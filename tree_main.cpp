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

void write_dataset(std::string filename, const Ref<const ArrayXXb>& dataset) {
    std::ofstream file;
    file.open(filename, std::ios::app);
    file << dataset;
    file.close();
}

void write_int_dataset(std::string filename, const Ref<const ArrayXXc>& dataset) {
    std::ofstream file;
    file.open(filename, std::ios::app);
    file << dataset;
    file.close();
}

int main(int argc, char *argv[]){
    omp_set_num_threads(23);
    Eigen::initParallel();

    bool mnist = 1;
    std::string dir_path = "/rigel/home/dmb2266/instance-optimal-lsh/"; 
    Dataset test_dataset_obj, train_dataset_obj;
    if (mnist) {
        std::string train_file = dir_path + "data/train-images-idx3-ubyte";

        std::cout << "Reading Data" << std::endl;
        train_dataset_obj.read_mnist(train_file);
    } else {
        std::cout << "ImageNet" << std::endl;
        std::string train_file = dir_path + "data/image_net_large.bin";

        train_dataset_obj.read_ImageNet(train_file);
    }
    

    std::cout << "Binarizing Data" << std::endl;
    unsigned char t=1;
    //test_dataset_obj.set_thresh(t);
    train_dataset_obj.set_thresh(t);
    //test_dataset_obj.binarize_data();
    train_dataset_obj.binarize_data();

    // Verify load 
    
    std::cout << "test Image" << std::endl;
    std::cout << train_dataset_obj.get_binary_data().row(0) << std::endl;


    std::cout << "Bucket size is ";
    std::cout << BUCKET_SIZE << std::endl; 

    //std::cout << "Query set size is " << test_dataset_obj.get_binary_data().rows() << std::endl;
    std::cout << "Data set size is " << train_dataset_obj.get_binary_data().rows() << std::endl;


    // init tree list
    std::vector<LSHtree*> tree_list; 

    // compute trees
    std::string tree_label = "";
    if (argc > 1) {
        tree_label = "tree";
        char* src = argv[1];
        int i=0;
        while (src[i] != '\0') {
            tree_label.push_back(src[i]);
            i++; 
        }
    } 
    
    std::string dfile = dir_path + "/dataset.txt"; 
    if (!fexists(dfile.c_str())) {
        write_dataset(dir_path + "/dataset.txt", train_dataset_obj.get_binary_data());
    }

    std::cout << tree_label << std::endl;


    // ANN Parameters
    int r=3;
    int c=1;

    bool build_trees = 1; 
    if (build_trees) {
        LSHtree* tree_ptr = compute_tree(r, c, &train_dataset_obj, tree_label);

        // write tree list to file
        std::string tree_file = dir_path + "trees/" + tree_label + "/hashes";
        std::ofstream file;
        file.open(tree_file, std::ofstream::app);
        std::cout << "writing to " << tree_file << std::endl;
        write_tree_file(tree_file, tree_ptr, file); 
        file.close();
    }

    // query test
    /*
    std::string test_dir = "/home/dmb2266/test_hashes/";
    std::system("move_hashes.sh");
    //std::system("prepare_test.sh");

    std::string prefix;
    if (mnist) {
        prefix = "MNIST";
    } else {
        prefix = "IN";
    }

    std::vector<std::string> hash_dirs;
    hash_dirs.push_back(test_dir + prefix + "_c1o2");
    hash_dirs.push_back(test_dir + prefix + "_uniform");

    //query_test(hash_dirs, &train_dataset_obj, r);
    */
    return 0; 
}
