#include "utils.h"
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <set>
using namespace std;
namespace po = boost::program_options;

// Split queries by ID file
// Reads an ID file (one ID per line) and splits queries into two files:
// - output_selected: queries whose indices are in the ID file
// - output_remaining: queries whose indices are NOT in the ID file
void split_queries_by_id(const float *queries, unsigned int nq, unsigned int dim,
                         const std::string &id_file_path,
                         const std::string &output_selected,
                         const std::string &output_remaining,
                         DataType dtype)
{
    // Read ID file
    std::set<unsigned int> selected_ids;
    std::ifstream id_file(id_file_path.c_str());
    if (!id_file.is_open())
    {
        std::cout << "Error: Cannot open ID file " << id_file_path << std::endl;
        return;
    }

    unsigned int id;
    while (id_file >> id)
    {
        if (id >= nq)
        {
            std::cout << "Warning: ID " << id << " exceeds query count " << nq << std::endl;
        }
        else
        {
            selected_ids.insert(id);
        }
    }
    id_file.close();

    std::cout << "Loaded " << selected_ids.size() << " IDs from " << id_file_path << std::endl;

    // Split queries
    std::vector<float> selected_queries;
    std::vector<float> remaining_queries;

    for (unsigned int i = 0; i < nq; i++)
    {
        if (selected_ids.count(i) > 0)
        {
            // This query is in the selected set
            for (unsigned int j = 0; j < dim; j++)
            {
                selected_queries.push_back(queries[i * dim + j]);
            }
        }
        else
        {
            // This query is in the remaining set
            for (unsigned int j = 0; j < dim; j++)
            {
                remaining_queries.push_back(queries[i * dim + j]);
            }
        }
    }

    unsigned int num_selected = selected_queries.size() / dim;
    unsigned int num_remaining = remaining_queries.size() / dim;

    // Save selected queries
    if (!output_selected.empty() && num_selected > 0)
    {
        save_data(output_selected, selected_queries.data(), num_selected, dim, dtype);
    }
    else if (num_selected == 0)
    {
        std::cout << "Warning: No selected queries to save" << std::endl;
    }

    // Save remaining queries
    if (!output_remaining.empty() && num_remaining > 0)
    {
        save_data(output_remaining, remaining_queries.data(), num_remaining, dim, dtype);
    }
    else if (num_remaining == 0)
    {
        std::cout << "Warning: No remaining queries to save" << std::endl;
    }

    std::cout << "Split result: " << num_selected << " selected queries, "
              << num_remaining << " remaining queries" << std::endl;
}

int main(int argc, char **argv)
{
    string query_path;
    string id_file_path;
    string output_selected;
    string output_remaining;
    string data_type_str;

    /***
     * Parameters reading
     ***/
    {
        po::options_description desc_visible("General options");
        desc_visible.add_options()
        ("help,h", "produce help message.")
        ("version,v", "print version information.")
        ("queries", po::value(&query_path)->required(), "input queries file path")
        ("id-file", po::value(&id_file_path)->required(), "input ID file path (one ID per line)")
        ("output-selected", po::value(&output_selected)->required(), "output file for selected queries")
        ("output-remaining", po::value(&output_remaining)->required(), "output file for remaining queries")
        ("data-type", po::value(&data_type_str)->default_value("float"), "data type: float, uint8, int8");

        po::options_description desc("Allowed options");
        desc.add(desc_visible);

        po::variables_map vm;
        try
        {
            po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

            if (vm.count("help"))
            {
                cout << "Query Splitter by ID\n\n";
                cout << desc_visible << "\n";
                cout << "Example usage:\n";
                cout << "  " << argv[0] << " --queries query.fbin --id-file ids.txt --output-selected selected.fbin --output-remaining remaining.fbin\n";
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
    {
        dtype = FLOAT32;
    }
    else if (data_type_str == "uint8")
    {
        dtype = UINT8;
    }
    else if (data_type_str == "int8")
    {
        dtype = INT8;
    }
    else
    {
        cout << "Error: Unknown data type '" << data_type_str << "'. Use float, uint8, or int8." << endl;
        return -1;
    }

    // Read query file header
    ifstream qin(query_path.c_str(), ios::binary);
    if (!qin.is_open())
    {
        cout << "Error: Cannot open query file " << query_path << endl;
        return -1;
    }

    int file_nq, file_dim_q;
    qin.read((char *)(&file_nq), sizeof(int));
    qin.read((char *)(&file_dim_q), sizeof(int));

    if (qin.fail())
    {
        cout << "Error: Failed to read query file header" << endl;
        qin.close();
        return -1;
    }

    qin.close();

    unsigned int num_queries = file_nq;
    unsigned int dim = file_dim_q;

    cout << "Query file: " << query_path << " (" << data_type_str << ")" << endl;
    cout << "Parameters: " << num_queries << " queries, " << dim << " dims" << endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Load queries
    float *queries = nullptr;
    load_data(query_path, queries, num_queries, dim, dtype);

    // Split queries by ID
    split_queries_by_id(queries, num_queries, dim, id_file_path, output_selected, output_remaining, dtype);

    delete[] queries;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    cout << "Total time: " << duration.count() << "s" << endl;

    return 0;
}
