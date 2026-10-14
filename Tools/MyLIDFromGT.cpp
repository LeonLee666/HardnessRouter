
// MyLIDFromGT.cpp
// Compute Local Intrinsic Dimensionality (LID) directly from a pre-computed
// Ground Truth (GT) file, skipping the expensive k-NN search step.
//
// GT file binary format (same as compute_groundtruth output):
//   [int32_t: npts ]     number of queries
//   [int32_t: ndims]     number of neighbors per query (= K)
//   [uint32_t: id_matrix[npts * ndims]]     K-NN IDs, row-major
//   [float:    dist_matrix[npts * ndims]]   K-NN squared-L2 distances, row-major,
//                                            sorted ascending per row
//
// LID (MLE estimator, identical to compute_lid in utils.h):
//   LID = -k / sum( log(d_i / d_k) )
// where d_i is the squared-L2 distance to the i-th nearest neighbor,
// and d_k is the distance to the k-th (farthest used) neighbor.

#include "utils.h"
#include <iomanip>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

// --------------------------------------------------------------------------
// Compute LID from a row of pre-computed k-NN distances.
//
// dists       : pointer to K distances for one query (sorted ascending)
// K_available : total neighbors available in the dists array
// k_use       : how many neighbors to actually use (<= K_available)
// dim         : data dimensionality (fallback when all distances are ~equal)
//
// Behavior matches compute_lid() in utils.h exactly, including all
// corner-case handling.
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// Compute LID from a row of pre-computed k-NN distances.
//
// dists       : pointer to K distances for one query, **sorted ascending**
// K_available : total neighbors available in the dists array
// k_use       : how many neighbors to actually use (<= K_available)
// dim         : data dimensionality (fallback when all distances are ~equal)
//
// Because dists is already sorted ascending, the farthest neighbor in the
// used range is simply dists[limit-1] — no scan needed.
//
// Behavior matches compute_lid() in utils.h exactly, including all
// corner-case handling.
// --------------------------------------------------------------------------
float compute_lid_from_dists(const float *dists,
                             unsigned int K_available,
                             unsigned int k_use,
                             unsigned int dim)
{
    unsigned int limit = min(K_available, k_use);
    if (limit == 0)
        return 0.0f;
    float maxdistk = dists[limit - 1];

    if (maxdistk == 0.0f)
        return 0.0f;

    double sum = 0;
    unsigned int actual_k = 0;

    for (unsigned int i = 0; i < limit; i++)
    {
        if (dists[i] == 0.0f)
            continue;

        sum += log((double)dists[i] / (double)maxdistk);
        actual_k++;
    }

    if (actual_k == 0)
        return 0.0f;

    if (sum == 0.0f)
    {
        return (float)dim;
    }

    return std::min((float)dim, (float)(-(double)actual_k / sum));
}

// --------------------------------------------------------------------------
// Load GT file: read header, detect format, load distance matrix only.
//
// Returns distance matrix via dist_matrix (heap-allocated, caller frees).
// Sets num_queries and K.  Returns true on success.
// --------------------------------------------------------------------------
bool load_gt_distances(const string &gt_path,
                       float *&dist_matrix,
                       unsigned int &num_queries,
                       unsigned int &K)
{
    ifstream in(gt_path.c_str(), ios::binary);
    if (!in.is_open())
    {
        cerr << "Error: Cannot open GT file " << gt_path << endl;
        return false;
    }

    // --- Read header ---
    int32_t npts = 0, ndims = 0;
    in.read((char *)(&npts), sizeof(int32_t));
    in.read((char *)(&ndims), sizeof(int32_t));

    if (in.fail() || npts <= 0 || ndims <= 0)
    {
        cerr << "Error: Invalid GT file header (npts=" << npts
             << ", ndims=" << ndims << ")" << endl;
        in.close();
        return false;
    }

    cout << "GT file: " << npts << " queries, " << ndims << " neighbors each" << endl;

    // --- Detect format by file size ---
    in.seekg(0, ios::end);
    streampos file_size = in.tellg();

    size_t hdr = 2 * sizeof(int32_t);
    size_t matrix_elems = (size_t)npts * (size_t)ndims;
    size_t expected_with_dist = hdr + 2 * matrix_elems * 4;  // IDs + distances
    size_t expected_id_only   = hdr + matrix_elems * 4;       // IDs only

    bool has_distances;
    if ((size_t)file_size == expected_with_dist)
    {
        has_distances = true;
    }
    else if ((size_t)file_size == expected_id_only)
    {
        has_distances = false;
    }
    else
    {
        cerr << "Error: GT file size mismatch." << endl;
        cerr << "  Actual file size:        " << file_size << " bytes" << endl;
        cerr << "  Expected (with dist):    " << expected_with_dist << " bytes" << endl;
        cerr << "  Expected (IDs only):     " << expected_id_only << " bytes" << endl;
        in.close();
        return false;
    }

    if (!has_distances)
    {
        cerr << "Error: GT file contains IDs only — no distance matrix." << endl;
        cerr << "       LID computation requires distances. Please provide a complete GT file." << endl;
        in.close();
        return false;
    }

    num_queries = (unsigned int)npts;
    K = (unsigned int)ndims;

    // --- Seek past the ID matrix to the distance matrix ---
    streamoff dist_offset = (streamoff)(hdr + matrix_elems * sizeof(uint32_t));
    in.clear();
    in.seekg(dist_offset, ios::beg);
    if (in.fail())
    {
        cerr << "Error: Failed to seek to distance matrix in GT file" << endl;
        in.close();
        return false;
    }

    // --- Read distance matrix ---
    size_t dist_count = matrix_elems;
    dist_matrix = new float[dist_count];
    in.read((char *)dist_matrix, (streamsize)(dist_count * sizeof(float)));
    bool read_ok = !in.fail();
    streamsize bytes_read = in.gcount();
    in.close();

    if (!read_ok || bytes_read != (streamsize)(dist_count * sizeof(float)))
    {
        cerr << "Error: Failed to read distance matrix (got " << bytes_read
             << " bytes, expected " << dist_count * sizeof(float) << ")" << endl;
        delete[] dist_matrix;
        dist_matrix = nullptr;
        return false;
    }

    // --- Quick sanity check on distance values ---
    float min_d = dist_matrix[0], max_d = dist_matrix[0];
    size_t check = min((size_t)1000, dist_count);
    for (size_t i = 1; i < check; i++)
    {
        if (dist_matrix[i] < min_d) min_d = dist_matrix[i];
        if (dist_matrix[i] > max_d) max_d = dist_matrix[i];
    }
    cout << "Distance range (sampled): min=" << min_d << ", max=" << max_d << endl;
    cout << "Successfully loaded distance matrix (" << dist_count << " floats)" << endl;

    return true;
}

int main(int argc, char **argv)
{
    string gt_path;
    string output_path;
    string dist_output_path;
    unsigned int k = 0;    // 0 = use all neighbors in GT
    unsigned int dim = 0;  // ambient dimensionality for corner-case fallback

    /***
     * Parameters
     ***/
    {
        po::options_description desc_visible("General options");
        desc_visible.add_options()
        ("help,h", "produce help message.")
        ("gt", po::value(&gt_path)->required(),
            "Ground Truth file (binary: npts, ndims, id_matrix[uint32], dist_matrix[float])")
        ("output", po::value(&output_path)->required(),
            "output file for LID values (one float per line)")
        ("dist-output", po::value(&dist_output_path)->default_value(""),
            "output file for per-query distances (one query per line, space-separated)")
        ("k", po::value(&k)->default_value(0),
            "number of nearest neighbors to use (0 = use all in GT file)")
        ("dim", po::value(&dim)->default_value(0),
            "data dimensionality, used as fallback when all distances are ~equal (0 = use K)");

        po::options_description desc("Allowed options");
        desc.add(desc_visible);

        po::variables_map vm;
        try
        {
            po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

            if (vm.count("help"))
            {
                cout << "LID Calculator from Ground Truth file\n\n";
                cout << desc_visible << "\n";
                cout << "Reads a pre-computed GT file (K-NN ids + distances) and computes\n";
                cout << "Local Intrinsic Dimensionality (LID) for each query.\n\n";
                cout << "Example usage:\n";
                cout << "  " << argv[0] << " --gt gt.bin --output lid.txt\n";
                cout << "  " << argv[0] << " --gt gt.bin --output lid.txt --k 50\n";
                cout << "  " << argv[0] << " --gt gt.bin --output lid.txt --k 100 --dim 128\n";
                return 0;
            }

            po::notify(vm);
        }
        catch (const po::error &e)
        {
            cerr << "Error: " << e.what() << "\n\n";
            cerr << desc_visible << "\n";
            return -1;
        }
    }

    /***
     * Load GT distances
     ***/
    float *dist_matrix = nullptr;
    unsigned int num_queries = 0;
    unsigned int K = 0;

    if (!load_gt_distances(gt_path, dist_matrix, num_queries, K))
        return -1;

    /***
     * Resolve parameters
     ***/
    unsigned int k_use = (k == 0) ? K : k;
    if (k_use > K)
    {
        cout << "Warning: requested k=" << k_use << " > neighbors in GT ("
             << K << "). Using K=" << K << "." << endl;
        k_use = K;
    }

    unsigned int dim_fallback = dim;
    if (dim_fallback == 0)
    {
        dim_fallback = K;   // rough upper bound; warn below
        cout << "Warning: --dim not specified; using K=" << K
             << " as fallback for degenerate cases." << endl;
        cout << "         Pass --dim <vector_length> for accurate handling." << endl;
    }

    cout << "Computing LID: " << num_queries << " queries, k=" << k_use << endl;

    /***
     * Compute LID for each query
     ***/
    auto start_time = chrono::high_resolution_clock::now();

    float *lidperquery = new float[num_queries];

    // Open distance output file (optional)
    ofstream dist_out;
    bool write_dist = !dist_output_path.empty();
    if (write_dist)
    {
        dist_out.open(dist_output_path.c_str());
        if (!dist_out.is_open())
        {
            cerr << "Error: Cannot open dist-output file " << dist_output_path << endl;
            delete[] lidperquery;
            delete[] dist_matrix;
            return -1;
        }
        dist_out << setprecision(10);
    }

    for (int i = 0; i < (int)num_queries; i++)
    {
        lidperquery[i] = compute_lid_from_dists(
            dist_matrix + (size_t)i * K, K, k_use, dim_fallback);
        if (lidperquery[i]==dim) {
            cout << "Warning: LID for query " << i << " is dim=" << dim << endl;
        }
        // Write this query's distances (k_use values, space-separated) to file
        if (write_dist)
        {
            const float *row = dist_matrix + (size_t)i * K;
            for (unsigned int j = 0; j < k_use; j++)
            {
                if (j > 0) dist_out << ',';
                dist_out << row[j];
            }
            dist_out << '\n';
        }
    }

    if (write_dist)
    {
        dist_out.close();
        cout << "Distances written to: " << dist_output_path << endl;
    }

    /***
     * Statistics
     ***/
    double sum_lid = 0;
    float min_lid = lidperquery[0];
    float max_lid = lidperquery[0];
    unsigned int zero_count = 0;   // queries that returned LID = 0 (corner case)
    unsigned int dim_count  = 0;   // queries that hit the sum≈0 fallback

    for (unsigned int i = 0; i < num_queries; i++)
    {
        sum_lid += lidperquery[i];
        if (lidperquery[i] < min_lid) min_lid = lidperquery[i];
        if (lidperquery[i] > max_lid) max_lid = lidperquery[i];
        if (lidperquery[i] == 0.0f) zero_count++;
        // Detect dim-fallback: value equals dim_fallback exactly
        if (lidperquery[i] == (float)dim_fallback && dim_fallback != 0) dim_count++;
    }
    float mean_lid = (float)(sum_lid / num_queries);

    // Median
    vector<float> sorted_lid(lidperquery, lidperquery + num_queries);
    sort(sorted_lid.begin(), sorted_lid.end());
    float median_lid = sorted_lid[num_queries / 2];

    cout << "\n--- LID Statistics ---" << endl;
    cout << "min    = " << min_lid << endl;
    cout << "max    = " << max_lid << endl;
    cout << "mean   = " << mean_lid << endl;
    cout << "median = " << median_lid << endl;
    if (zero_count > 0)
        cout << "corner cases: " << zero_count << " queries returned LID=0 "
             << "(no valid neighbors or all-zero distances)" << endl;
    if (dim_count > 0)
        cout << "corner cases: " << dim_count << " queries hit sum≈0 fallback "
             << "(returned dim=" << dim_fallback << ")" << endl;

    /***
     * Write output
     ***/
    ofstream outfile(output_path.c_str());
    if (!outfile.is_open())
    {
        cerr << "Error: Cannot open output file " << output_path << endl;
        delete[] dist_matrix;
        delete[] lidperquery;
        return -1;
    }
    // Use higher precision for output
    outfile << setprecision(10);
    for (unsigned int i = 0; i < num_queries; i++)
    {
        outfile << lidperquery[i] << "\n";
    }
    outfile.close();
    cout << "LID values written to: " << output_path << endl;

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "\nAverage LID: " << mean_lid << endl;
    cout << "Total time: " << duration.count() << "ms" << endl;

    delete[] dist_matrix;
    delete[] lidperquery;
    return 0;
}
