// QueryNoiseAugment.cpp
// Read a query file (binary: npts, ndims, matrix), and for each query vector
// generate N new vectors by adding Gaussian noise N(0, sigma^2).
// Output file has the same binary format, with npts*N vectors.
//
// Supports float32 / uint8 / int8 data types.
// Fixed random seed for reproducibility (deterministic across runs and
// independent of thread count).
//
// OpenMP parallelized.

#include "utils.h"
#include <omp.h>
#include <boost/program_options.hpp>
#include <random>
#include <cmath>
using namespace std;
namespace po = boost::program_options;

// Fixed base seed for reproducibility
static const uint32_t BASE_SEED = 42;

// --------------------------------------------------------------------------
// Clamp a float value to the integer range [lo, hi] and round to nearest.
//
// Clamping MUST happen before the cast: without it, values just outside the
// range would wrap around (e.g. 127+1 -> -128), creating a huge spurious
// perturbation. Also note "!(val >= lo)" instead of "val < lo" so that NaN
// is caught too (NaN compares false against everything); casting NaN to an
// integer is undefined behavior and can likewise produce a wild value.
// --------------------------------------------------------------------------
template <typename T>
inline T clamp_round(float val, float lo, float hi)
{
    if (!(val >= lo)) val = lo;   // catches val < lo and NaN
    if (val > hi) val = hi;       // catches overflow
    return static_cast<T>(roundf(val));
}

// --------------------------------------------------------------------------
// Deterministic per-vector seed: each (query, copy) pair maps to a unique
// RNG state, so output is identical regardless of thread count or ordering.
// --------------------------------------------------------------------------
inline uint32_t make_seed(unsigned int query_idx, unsigned int copy_idx)
{
    // Simple mixing — good enough for distinct RNG streams
    return BASE_SEED + query_idx * 1000003u + copy_idx;
}

// --------------------------------------------------------------------------
// Pick a sensible default sigma based on data type.
//   float : 0.01  — tiny perturbation, preserved in float precision
//   uint8 : 1.0   — smallest meaningful integer perturbation (±1 per dim)
//   int8  : 1.0   — same
// --------------------------------------------------------------------------
float default_sigma_for_type(DataType dtype)
{
    switch (dtype)
    {
    case FLOAT32: return 0.01f;
    case UINT8:   return 1.0f;
    case INT8:    return 0.5f;
    default:      return 1.0f;
    }
}

// --------------------------------------------------------------------------
// Generate noisy vectors for all queries (OpenMP parallel).
//
// queries     : input vectors, shape [nq * dim], stored as float
// nq          : number of input queries
// dim         : vector dimensionality
// num_copies  : how many noisy copies per original query
// sigma       : Gaussian noise standard deviation (in float space)
// dtype       : output data type (FLOAT32 / UINT8 / INT8)
// output_path : where to write the augmented file
// --------------------------------------------------------------------------
void augment_with_noise(const float *queries,
                        unsigned int nq,
                        unsigned int dim,
                        unsigned int num_copies,
                        float sigma,
                        DataType dtype,
                        const string &output_path)
{
    unsigned int total_out = nq * num_copies;
    cout << "Generating " << total_out << " vectors ("
         << nq << " x " << num_copies << ")" << endl;
    cout << "Noise: Gaussian N(0, " << sigma << "^2), seed=" << BASE_SEED << endl;
    cout << "Using " << omp_get_max_threads() << " threads" << endl;

    // --- Pre-allocate output buffer in memory ---
    size_t elem_size = (dtype == FLOAT32) ? sizeof(float) : 1;
    size_t total_bytes = (size_t)total_out * dim * elem_size;

    // --- Parallel generation into memory buffer ---
    // Each (query, copy) pair gets its own deterministic seed, so the output
    // is identical regardless of how many threads are used.
    if (dtype == FLOAT32)
    {
        float *out_buf = new float[(size_t)total_out * dim];

        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < (int)nq; q++)
        {
            const float *src = queries + (size_t)q * dim;
            for (unsigned int c = 0; c < num_copies; c++)
            {
                mt19937 rng(make_seed(q, c));
                normal_distribution<float> gauss(0.0f, sigma);

                float *dst = out_buf + ((size_t)q * num_copies + c) * dim;
                for (unsigned int d = 0; d < dim; d++)
                {
                    dst[d] = src[d] + gauss(rng);
                }
            }
        }

        // --- Write to file ---
        ofstream out(output_path.c_str(), ios::binary);
        if (!out.is_open())
        {
            cerr << "Error: Cannot open output file " << output_path << endl;
            delete[] out_buf;
            return;
        }
        int hdr_npts = (int)total_out;
        int hdr_dim  = (int)dim;
        out.write((const char *)&hdr_npts, sizeof(int));
        out.write((const char *)&hdr_dim,  sizeof(int));
        out.write((const char *)out_buf, (streamsize)total_bytes);
        out.close();
        delete[] out_buf;
    }
    else if (dtype == UINT8)
    {
        uint8_t *out_buf = new uint8_t[(size_t)total_out * dim];

        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < (int)nq; q++)
        {
            const float *src = queries + (size_t)q * dim;
            for (unsigned int c = 0; c < num_copies; c++)
            {
                mt19937 rng(make_seed(q, c));
                normal_distribution<float> gauss(0.0f, sigma);

                uint8_t *dst = out_buf + ((size_t)q * num_copies + c) * dim;
                for (unsigned int d = 0; d < dim; d++)
                {
                    dst[d] = clamp_round<uint8_t>(src[d] + gauss(rng), 0.0f, 255.0f);
                }
            }
        }

        ofstream out(output_path.c_str(), ios::binary);
        if (!out.is_open())
        {
            cerr << "Error: Cannot open output file " << output_path << endl;
            delete[] out_buf;
            return;
        }
        int hdr_npts = (int)total_out;
        int hdr_dim  = (int)dim;
        out.write((const char *)&hdr_npts, sizeof(int));
        out.write((const char *)&hdr_dim,  sizeof(int));
        out.write((const char *)out_buf, (streamsize)total_bytes);
        out.close();
        delete[] out_buf;
    }
    else // INT8
    {
        int8_t *out_buf = new int8_t[(size_t)total_out * dim];

        // Count dimensions whose noise would leave the int8 range before
        // clamping. With a small sigma this should be ~0; if it is large,
        // the clamp is silently shrinking perturbations at the boundary.
        long long n_oob = 0;

        #pragma omp parallel for schedule(dynamic) reduction(+:n_oob)
        for (int q = 0; q < (int)nq; q++)
        {
            const float *src = queries + (size_t)q * dim;
            for (unsigned int c = 0; c < num_copies; c++)
            {
                mt19937 rng(make_seed(q, c));
                normal_distribution<float> gauss(0.0f, sigma);

                int8_t *dst = out_buf + ((size_t)q * num_copies + c) * dim;
                for (unsigned int d = 0; d < dim; d++)
                {
                    float noisy = src[d] + gauss(rng);
                    if (noisy < -128.0f || noisy > 127.0f)
                        n_oob++;
                    dst[d] = clamp_round<int8_t>(noisy, -128.0f, 127.0f);
                }
            }
        }

        // Report how much the clamp actually had to truncate
        size_t total_dims = (size_t)total_out * dim;
        if (n_oob > 0)
            cout << "Clamp: " << n_oob << " of " << total_dims << " dims ("
                 << 100.0 * n_oob / total_dims
                 << "%) left the int8 range and were truncated" << endl;
        else
            cout << "Clamp: no dimension left the int8 range; "
                 << "all perturbations kept in full" << endl;

        ofstream out(output_path.c_str(), ios::binary);
        if (!out.is_open())
        {
            cerr << "Error: Cannot open output file " << output_path << endl;
            delete[] out_buf;
            return;
        }
        int hdr_npts = (int)total_out;
        int hdr_dim  = (int)dim;
        out.write((const char *)&hdr_npts, sizeof(int));
        out.write((const char *)&hdr_dim,  sizeof(int));
        out.write((const char *)out_buf, (streamsize)total_bytes);
        out.close();
        delete[] out_buf;
    }

    // --- Verify file size ---
    size_t expected = 8 + total_bytes;
    ifstream check(output_path.c_str(), ios::binary | ios::ate);
    if (check.is_open())
    {
        streampos actual = check.tellg();
        check.close();
        if ((size_t)actual == expected)
            cout << "Output file size OK: " << actual << " bytes" << endl;
        else
            cerr << "Warning: output file size " << actual
                 << " != expected " << expected << " bytes" << endl;
    }

    cout << "Done. " << total_out << " vectors written to: " << output_path << endl;
}

int main(int argc, char **argv)
{
    string query_path;
    string output_path;
    string data_type_str;
    unsigned int num_copies = 10;
    double noise_sigma = -1.0;   // -1 = auto (pick based on data type)

    /***
     * Parameters
     ***/
    {
        po::options_description desc_visible("General options");
        desc_visible.add_options()
        ("help,h", "produce help message.")
        ("queries", po::value(&query_path)->required(),
            "input query file (binary: npts, ndims, matrix)")
        ("output", po::value(&output_path)->required(),
            "output augmented query file (same format)")
        ("num-copies", po::value(&num_copies)->default_value(10),
            "number of noisy copies per original query (default: 10)")
        ("noise-sigma", po::value(&noise_sigma)->default_value(-1.0),
            "Gaussian noise std (-1 = auto: 0.01 for float, 1.0 for uint8/int8)")
        ("data-type", po::value(&data_type_str)->default_value("float"),
            "data type: float, uint8, int8");

        po::options_description desc("Allowed options");
        desc.add(desc_visible);

        po::variables_map vm;
        try
        {
            po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

            if (vm.count("help"))
            {
                cout << "Query Noise Augmentation\n\n";
                cout << "For each input query, generate N nearby points by adding\n";
                cout << "Gaussian noise N(0, sigma^2). Output has npts*N vectors.\n\n";
                cout << desc_visible << "\n";
                cout << "Example usage:\n";
                cout << "  " << argv[0]
                     << " --queries query.fbin --output aug_query.fbin\n";
                cout << "  " << argv[0]
                     << " --queries query.fbin --output aug_query.fbin"
                     << " --num-copies 20 --noise-sigma 2.0 --data-type uint8\n";
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

    // Parse data type
    DataType dtype = FLOAT32;
    if (data_type_str == "float" || data_type_str == "float32")
        dtype = FLOAT32;
    else if (data_type_str == "uint8")
        dtype = UINT8;
    else if (data_type_str == "int8")
        dtype = INT8;
    else
    {
        cerr << "Error: Unknown data type '" << data_type_str
             << "'. Use float, uint8, or int8." << endl;
        return -1;
    }

    if (num_copies == 0)
    {
        cerr << "Error: --num-copies must be > 0" << endl;
        return -1;
    }

    // Resolve sigma: auto-pick if not specified
    float sigma;
    if (noise_sigma < 0)
    {
        sigma = default_sigma_for_type(dtype);
        cout << "Auto sigma for " << data_type_str << ": " << sigma << endl;
    }
    else
    {
        sigma = (float)noise_sigma;
    }

    /***
     * Read query file header
     ***/
    ifstream qin(query_path.c_str(), ios::binary);
    if (!qin.is_open())
    {
        cerr << "Error: Cannot open query file " << query_path << endl;
        return -1;
    }
    int file_nq = 0, file_dim = 0;
    qin.read((char *)&file_nq,  sizeof(int));
    qin.read((char *)&file_dim, sizeof(int));
    if (qin.fail() || file_nq <= 0 || file_dim <= 0)
    {
        cerr << "Error: Invalid query file header (npts=" << file_nq
             << ", dim=" << file_dim << ")" << endl;
        qin.close();
        return -1;
    }
    qin.close();

    unsigned int num_queries = (unsigned int)file_nq;
    unsigned int dim         = (unsigned int)file_dim;

    cout << "Input: " << query_path << " (" << data_type_str << ")" << endl;
    cout << "  " << num_queries << " queries, " << dim << " dims" << endl;
    cout << "Output: " << num_queries * num_copies << " queries ("
         << num_copies << "x augmentation)" << endl
         << endl;

    auto start_time = chrono::high_resolution_clock::now();

    /***
     * Load queries (load_data converts everything to float internally)
     ***/
    float *queries = nullptr;
    load_data(query_path, queries, num_queries, dim, dtype);

    /***
     * Augment
     ***/
    augment_with_noise(queries, num_queries, dim,
                       num_copies, sigma, dtype, output_path);

    delete[] queries;

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total time: " << duration.count() << "ms" << endl;

    return 0;
}
