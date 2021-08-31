#ifndef __TEST_H
#define __TEST_H

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


void query_test(std::vector<std::string>, Dataset*, int);


#endif