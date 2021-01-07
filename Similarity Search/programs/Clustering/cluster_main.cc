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
	// construct the cluster class depending on which algorithm the user has chosen
	instant_reduced = new Clustering<double>("lloyds", feature_vectors_reduced, k);
	instant = new Clustering<double>("lloyds", feature_vectors, k);
	
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
	
	// // compute the silhouette
	// auto pair = instant_reduced->compute_silhouette();
	// // print the silhouette results
	// output << "\nSilhouette: [";
	// print::vector_print_infile(pair.first, &output);
	// output << ", " << pair.second << "]\n\n";


	// print stats for each cluster
	for (int i = 0; i < k; i++) {
		output << "CLUSTER-" << i << " {size: " << centroids_map.at(i).size() << ", centroid: ";
		print::vector_print_infile(centroids.at(i), &output);
		output << "}\n";
	}

	// print the clustering time
	output << "clustering_time: " <<  cluster_time.count() << endl;
	
	// // compute the silhouette
	// auto pair = instant->compute_silhouette();
	// // print the silhouette results
	// output << "\nSilhouette: [";
	// print::vector_print_infile(pair.first, &output);
	// output << ", " << pair.second << "]\n\n";

	// free the class pointer to wrap it up
	
	delete instant;

	output.close();

	free(input_file);
	free(config_file);
	free(output_file);
}