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
#include "../../algorithms/Clustering/clustering.h"

using namespace std::chrono;


int main(int argc, char* argv[]) {
	// parse the arguments
	char* input_file = NULL, *config_file = NULL, *output_file = NULL, *input_file_reduced = NULL, *clusters_from_NN = NULL;

	parse_clustering_args(argc, argv, &input_file, &output_file, &config_file, &input_file_reduced, &clusters_from_NN);

	// get the clustering parameters from the config file
	int k; 

	parse_clustering_config(config_file, &k);
	cout << "Parsing the files" << endl;
	// parse the dataset in order to get a vector of our feature vector
	vector<vector<double>> feature_vectors = parse_input(input_file, false);
	vector<vector<double>> feature_vectors_reduced = parse_input(input_file_reduced, true);

	// create the output file that we are going to use to print the clustering results
	ofstream output; 
	output.open(output_file);

	Clustering<double>* instant;
	Clustering<double>* instant_reduced;
	Clustering<double>* instant_nn;

	// construct the cluster class depending on which algorithm the user has chosen
	instant_reduced = new Clustering<double>("lloyds", feature_vectors_reduced, feature_vectors, k);
	instant = new Clustering<double>("lloyds", feature_vectors, k);

	
	cout << "Creating Clusters from the NN file" << endl;
	// create the clusters from the NN file. The constructor creates ecerything, no need for running the algorithm
	instant_nn = new Clustering<double>(feature_vectors, clusters_from_NN, k);

	cout << "Running clustering algorithm for the original space" << endl;
	auto c_start = std::chrono::high_resolution_clock::now();
	// run the clustering algorithm
	instant->run_clustering();
	auto c_end = std::chrono::high_resolution_clock::now();

	duration<double> cluster_time = duration_cast<duration<double>>(c_end - c_start);  
	
	// get the map of the centroids
	vector<vector<int>> centroids_map = instant->get_centroids_map();
	// get the actual centroids
	vector<vector<double>> centroids = instant->get_centroids();
	
	cout << "Running clustering algorithm for the reduced space" << endl;
	// same for the reduced vectors
	auto r_start = std::chrono::high_resolution_clock::now();
	// run the clustering algorithm
	instant_reduced->run_clustering();
	auto r_end = std::chrono::high_resolution_clock::now();

	duration<double> cluster_red_time = duration_cast<duration<double>>(r_end - r_start);  
	
	// get the map of the centroids
	vector<vector<int>> reduced_centroids_map = instant_reduced->get_centroids_map();
	// get the actual centroids
	vector<vector<double>> reduced_centroids = instant_reduced->get_centroids();

	cout << "Computing silhouettes for each architecture" << endl;
	// compute the silhouette for the reduced space
	auto red_pair = instant_reduced->compute_silhouette();
	auto pair_nn = instant_nn->compute_silhouette();
	auto pair = instant->compute_silhouette();

	// reduced vectors' stats
	// print stats for each cluster
	output << "NEW SPACE" << endl;
	for (int i = 0; i < k; i++) {
		output << "CLUSTER-" << i << " {size: " << reduced_centroids_map.at(i).size() << ", centroid: ";
		print::vector_print_infile(reduced_centroids.at(i), &output);
		output << "}\n";
	}
	// print the clustering time
	output << "clustering_time: " <<  cluster_red_time.count() << endl;

	// print the silhouette results
	output << "\nSilhouette: [";
	print::vector_print_infile(red_pair.first, &output);
	output << ", " << red_pair.second << "]\n\n";
	output << "Value of Objective Function: " << instant_reduced->compute_objective_function() << endl << endl;


	output << "ORIGINAL SPACE" << endl;
	// print stats for each cluster
	for (int i = 0; i < k; i++) {
		output << "CLUSTER-" << i << " {size: " << centroids_map.at(i).size() << ", centroid: ";
		print::vector_print_infile(centroids.at(i), &output);
		output << "}\n";
	}

	// print the clustering time
	output << "clustering_time: " <<  cluster_time.count() << endl;
	// print the silhouette results
	output << "\nSilhouette: [";
	print::vector_print_infile(pair.first, &output);
	output << ", " << pair.second << "]\n\n";
	output << "Value of Objective Function: " << instant->compute_objective_function() << endl << endl;


	output << "CLASSES AS CLUSTERS" << endl;
	// print the silhouette results
	output << "\nSilhouette: [";
	print::vector_print_infile(pair_nn.first, &output);
	output << ", " << pair_nn.second << "]\n\n";
	output << "Value of Objective Function: " << instant_nn->compute_objective_function() << endl << endl;


	// free the class pointer to wrap it up	
	delete instant;
	delete instant_nn;
	delete instant_reduced;

	output.close();

	free(input_file);
	free(input_file_reduced);
	free(config_file);
	free(output_file);
	free(clusters_from_NN);
}