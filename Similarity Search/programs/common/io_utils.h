#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <list> 
#include <vector> 

#include "../../utils/math_utils.h"

std::vector<std::vector<double>> parse_input(std::string filename, bool reduced);
void parse_lsh_args(int argc, char* argv[], char** input, char** output, char** query, char** query_low, char** input_low, int* k, int* L, int* n_neighs);
void parse_hc_args(int argc, char* argv[], char** input, char** output, char** query, int* k, int* M, int* probes, int* n_neighs, int* radius);
void parse_clustering_args(int argc, char* argv[], char** input, char** output, char** config, char** input_file_reduced, char** clusters_from_NN);
void parse_clustering_config(char* file, int* k);