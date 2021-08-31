#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "mnist_loader.h"
#include "utils.h"
#include "tree_comp.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <omp.h>
#include <chrono>

using namespace Eigen; 

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

bool query_file(std::string hash_file, const Ref<const ArrayXb>& query, int r) {
    
}


double timed_query(std::string param_dir, const Ref<const ArrayXb>& query, int r) {
    int num_hashes = 100;
    bool found = 0;

    // compute time needed to find near neighbor (<r in distance)
    auto t1 = high_resolution_clock::now();
    for (int i=0; i<num_hashes; i++) {
        std::string hash_file = param_dir + "hash" + std::to_string(i); 
        found = query_file(hash_file, query, r);
        if (found) {
            break;
        }
    }
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    double time = ms_double.count();

    return time;
}

void compute_times(std::string param_dir, std::vector<ArrayXb> queries, std::vector<double> times, int r)
{
    for (int i=0; i<queries.size(); i++) {
        double time;
        time = timed_query(param_dir, &queries.at(i), r)
        times.push_back(times)
    }
}

void write_times_to_file(std::string param_dir, std::vector<double> times) 
{
    std::string filename = param_dir + "times.txt";
    std::ofstream file;
    file.open(filename, std::ios::app);
    file << times;
    file.close()
}

void query_test(std::vector<std::string> hash_dirs, Dataset* dataset_ptr, int r) 
{
    std::vector<ArrayXb> queries = generate_queries(dataset_ptr);
    //std::vector<std::vector<double>> compare_times;
    for (int i=0; i<hash_dirs.size(), i++) {
        std::vector<double> times;
        std::string param_dir = hash_dirs.at(i);
        compute_times(param_dir, &queries, &times, r);
        write_times_to_file(param_dir, &times);
    }
}