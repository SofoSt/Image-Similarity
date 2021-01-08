#include <iostream>
#include <fstream>
#include <string>
#include <list> 
#include <vector> 
#include <cstring>
#include <assert.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <assert.h>

#include "../../utils/math_utils.h"
#include "../../utils/print_utils.h"
#include "../common/io_utils.h"
#include "../../algorithms/Search/LSH/headers/lsh.h"
#include "../../algorithms/Search/BruteForce/headers/brute_force.h"

using namespace std::chrono;

int main(int argc, char* argv[]) {
	// first step: parse the arguments
	char* input_file = NULL, *query_file = NULL, *output_file = NULL, *query_low_dim = NULL, *input_low_dim = NULL;
	int k, L, n_neighbors;
	double sum1 = 0, sum2 = 0, sum3 = 0;
	double aprox_lsh = 0, aprox_nn = 0;
	// parse the comand line arguments given
	parse_lsh_args(argc, argv, &input_file, &output_file, &query_file,
	&query_low_dim, &input_low_dim, &k, &L, &n_neighbors);

	// parse the dataset in order to get a vector of our feature vector
	vector<vector<double>> feature_vectors = parse_input(input_file, false);

	// parse the query set in order to find all of the queries that the user wants to ask
	vector<vector<double>> query_vectors = parse_input(query_file, false);

	// parse the dataset in order to get a vector of our low dimension feature vector
	vector<vector<double>> low_dim_feature_vectors = parse_input(input_low_dim, true);

	// parse the query set in order to find all of the low dimension queries that the user wants to ask
	vector<vector<double>> low_dim_query_vectors = parse_input(query_low_dim, true);

	// make sure that there is at least one element in the dataset and the queryset
	assert(feature_vectors.size() && query_vectors.size() && low_dim_feature_vectors.size() && low_dim_query_vectors.size());
	
	// create the output file that we are going to use to print the LSH results
	ofstream output;
	output.open(output_file);

	// lsh initialization values that we've learned from our dataset
	int n_points = feature_vectors.size();
	int true_space_dimension = feature_vectors.at(1).size();
	int reduced_space_dimension = low_dim_feature_vectors.at(1).size();
	int queries = query_vectors.size();
	// defalt lsh values, that we've learned for theory
	uint64_t m = pow(2,32) - 5;
	uint32_t M = 256;
	uint32_t w = our_math::compute_w_value(feature_vectors, 1000) / 4;

	// initialize our LSH class with the deature vectors and the aprropriate given values
	LSH<double> lsh_instant(L, m, M, n_points, k, true_space_dimension, w, feature_vectors);

	// initialize our brute force class, in order to find the true distance of the query and its neighbors
	BruteForce<double> bf_instant(n_points, true_space_dimension, feature_vectors);

	// initialize our brute force class for the reduced vectors' space
	BruteForce<double> red_bf_instant(n_points, reduced_space_dimension, low_dim_feature_vectors);
	
	// run all the queries with LSH
	for (uint64_t i = 0; i < (uint64_t)queries; i++) {
		// output the query number in the file
		output << "Query: " << i << endl;

		auto k_start = std::chrono::high_resolution_clock::now();
		// run the k-nearest neighbor algorithm, in order to obtain n neighbors
		pair<int,double> kNN = lsh_instant.NearestNeighbour(query_vectors.at(i));
		auto k_end = std::chrono::high_resolution_clock::now();

		duration<double> knn_time_span = duration_cast<duration<double>>(k_end - k_start);  
		
		auto b_start = std::chrono::high_resolution_clock::now();
		// run the brute force algorithm, in order to obtain the true nearest neighbors
		pair<int, double> kBF = bf_instant.RunBruteForce(query_vectors.at(i));
		auto b_end = std::chrono::high_resolution_clock::now();

		duration<double> bf_time_span = duration_cast<duration<double>>(b_end - b_start);  

		auto r_start = std::chrono::high_resolution_clock::now();
		// run the brute force algorithm, in order to obtain the true nearest neighbors
		pair<int, double> kRED = red_bf_instant.RunBruteForce(low_dim_query_vectors.at(i));
		auto r_end = std::chrono::high_resolution_clock::now();

		duration<double> red_time_span = duration_cast<duration<double>>(r_end - r_start);  

		// output the findings of kNN the Brute Force and the reduced space search
		output << "Nearest neighbor Reduced: " << kRED.first << endl;
		output << "Nearest neighbor LSH: " << kNN.first << endl;
		output << "Nearest neighbor True: " << kBF.first << endl;

		double distance_reduced = metrics::ManhatanDistance(query_vectors.at(i), feature_vectors.at(kRED.first), true_space_dimension);

		output << "DistanceReduced:" << distance_reduced << endl;
		output << "DistanceLSH:" << kNN.second << endl;
		output << "DistanceTrue:" << kBF.second << endl;
		
		aprox_lsh += metrics::ManhatanDistance(feature_vectors.at(kNN.first), query_vectors.at(i), true_space_dimension) / metrics::ManhatanDistance(feature_vectors.at(kBF.first), query_vectors.at(i), true_space_dimension);
		aprox_nn += distance_reduced / metrics::ManhatanDistance(feature_vectors.at(kBF.first), query_vectors.at(i), true_space_dimension);

		sum1 += knn_time_span.count();
		sum2 += bf_time_span.count();
		sum3 += red_time_span.count();
		
		output << "\n\n\n";
	}
	cout << "tLSH: " << sum1 / queries << endl;
	cout << "tBF: " << sum2 / queries << endl;
	cout << "tRED: " << sum3 / queries << endl;
	cout << "Approximation Factor LSH: " << aprox_lsh / queries << endl;
	cout << "Approximation Factor Reduced: " << aprox_nn / queries << endl;

	output.close();
	
	free(input_file);
	free(query_file);
	free(input_low_dim);
	free(query_low_dim);
	free(output_file);

	return 0;
}