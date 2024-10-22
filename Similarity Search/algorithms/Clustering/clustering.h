#pragma once

#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <ctime>
#include <utility>
#include <fstream>
#include <string>
#include <sstream>

#include "../Search/LSH/headers/lsh.h"
#include "../Search/Hypercube/headers/hypercube.h"

#define UNASSIGNED -1

template <typename T>
class Clustering {
	private:
		int k; // number of clusters needed
		int n_points; //number of points in the dataset
		uint32_t space_dim; // space dimension
		uint32_t original_space_dim; // space dimension

		bool using_reduced_space = false;
		/** assignment method chosen by the user. Options are:
		  		- lloyds
				- reverse_LSH
				- reverse_Hypercube 
		 **/

		std::string assignment_method;
		
		// vectors of the centroids
		std::vector<std::vector<T>> centroids;
		// vectors of the centroids
		std::vector<std::vector<T>> original_centroids;
		
		// vector of the vectors given to us, stored here for convinience
		std::vector<std::vector<T>> feature_vectors;
		// vector of the vectors given to us, stored here for convinience
		std::vector<std::vector<T>> original_feature_vectors;


		// vector that hols the index of the centroid that the current index's vector is assigned to
		std::vector<int> assigned_centroid;

		// in case of lsh, we need these extra variables
		uint32_t lsh_l;
		uint32_t lsh_k;
		LSH<T>* lsh_instant;

		// in case of hypercube, we need these extra variables
		uint32_t hc_M;
		uint32_t hc_k;
		uint32_t hc_probes;
		Hypercube<T>* hc_instant;

		//compute the nearest center for a given vector
		int nearest_centroid(vector<T> vec) {
			T min_dist = (T)INT_MAX;
			int nearest = -1;
			// compute the distances to all the centroids
			for (int i = 0; i < k; i++) {
				T dist = metrics::ManhatanDistance(vec, centroids.at(i), space_dim);
				// set it as min
				if (dist < min_dist) {
					min_dist = dist;
					nearest = i;
				}
			}
			assert(nearest != -1);

			return nearest;
		};

		// compute the second nearest centroid of a vector. Needed for the silhouette algorithm
		int second_nearest_centroid(vector <T> vec) {
			// get the nearest centroid
			int nearest = nearest_centroid(vec);

			T min_dist = (T)INT_MAX;
			int second_nearest = -1;
			// compute the distances to all the centroids
			for (int i = 0; i < k; i++) {
				T dist = metrics::ManhatanDistance(vec, centroids.at(i), space_dim);
				// set it as min
				if (dist < min_dist && i != nearest) {
					min_dist = dist;
					second_nearest = i;
				}
			}
			assert(second_nearest != -1);

			return second_nearest;
		};

		// compute how many unassigned vectors we've got, for the reverse assignement method
		int count_unassigned(vector<int> assigned) {
			int res = 0;
			for (int i = 0; i < n_points; i++) {
				if (assigned.at(i) == UNASSIGNED) {
					res++;
				}
			}
			
			return res;
		}

		// compute the minimum of the distances of the centroids. Needed for the initialization of the radius in reverse assignment
		T min_dist_between_centroids() {
			// initialize the min distance
			T min_dist = (T)INT_MAX;
			// brute force all the distances in order to find the smallest
			for (int i = 0; i < k; i++) {
				for (int j = 0; j < k; j++) {
					if (i != j) {
						T dist = metrics::ManhatanDistance(centroids.at(i), centroids.at(j), space_dim);
						if (dist < min_dist)
							min_dist = dist;
					}
				}
			} 

			return min_dist;
		}

		// parse the python input file
		void parser(string file_name, int k) {
			// we need a big line, in order to read all of our data
			string line[k];
			for (int i = 0; i < k; i++)
				line[i].resize(100000);
			// keep track of the clusters
			int cluster_counter = 0;

			// open the file
			ifstream input_file(file_name);
			if (input_file.is_open()) {
				// read each line(every line is one cluster)
				while (getline(input_file, line[cluster_counter])) {
					// take the first substring, which is then being ignored
					stringstream ss(line[cluster_counter]);
					string str;
					getline(ss, str, ',');

					// read each vector id
					while (ss.good()) {
						string sub_str;
						// vectors are seperated by commas
						getline(ss, sub_str, ',');
						// until we reach the end of the string
						if ( sub_str.find(" }") == string::npos) {
							int current = stoi(sub_str);
							// update the assigned centroids map
							assigned_centroid[current] = cluster_counter;
						} 
						
					}
					// break if we've read all the clusters
					if (++cluster_counter  == k)
						break;
				}
			}
			// close the file
			input_file.close();
		}	

	public:
		// constructor when original space is requested
		Clustering(std::string assignment_method, std::vector<std::vector<T>> feature_vectors, uint32_t k)
		: feature_vectors(feature_vectors), k(k) {
			if (assignment_method != "lloyds") {
				cout << "Wrong assignment method selected\n";
				exit(EXIT_FAILURE);
			}
			// initialize all the classs fields necessary
			this->assignment_method = assignment_method;
			n_points = feature_vectors.size();
			assert(n_points);
			space_dim = feature_vectors.at(0).size();

			vector<int> assigned(n_points, 0);
			assigned_centroid = assigned;

		};
		// constructor when reduced space is requested
		Clustering(std::string assignment_method, std::vector<std::vector<T>> feature_vectors, std::vector<std::vector<T>> org_feature_vectors, uint32_t k)
		: feature_vectors(feature_vectors), original_feature_vectors(org_feature_vectors), k(k) {
			if (assignment_method != "lloyds") {
				cout << "Wrong assignment method selected\n";
				exit(EXIT_FAILURE);
			}
			// initialize all the classs fields necessary
			this->assignment_method = assignment_method;
			n_points = feature_vectors.size();
			assert(n_points);
			space_dim = feature_vectors.at(0).size();
			original_space_dim = original_feature_vectors.at(0).size();

			vector<int> assigned(n_points, 0);
			assigned_centroid = assigned;

			using_reduced_space = true;

		};

		// constructor when reading the clusters from a file
		Clustering(std::vector<std::vector<T>> feature_vectors, string file_name, int k):
			feature_vectors(feature_vectors), k(k) {

				n_points = feature_vectors.size();
				assert(n_points);

				vector<int> assigned(n_points, 0);
				assigned_centroid = assigned;
				space_dim = feature_vectors.at(0).size();
				// parse the file in order to obtain the assigned centroid for each vector
				parser(file_name, k);
				// create the centroids from the info that we already have
				for (int i = 0; i < k; i++) {
					vector<T> temp_centroid(space_dim, 0);
					centroids.push_back(temp_centroid);
				}
				// update the centroids to get the median
				update();
			}

		// destructor for the clustering class
		~Clustering(){
			// free the pointers to the classes in case we have a reverse assignment method selected
			if (assignment_method == "reverse_LSH") 
				delete lsh_instant;
			if (assignment_method == "reverse_Hypercube") 
				delete hc_instant;

		};

		// get the vector of the centroids
		std::vector<std::vector<T>> get_centroids() {
			return this->centroids;
		};

		std::vector<std::vector<int>> get_centroids_map() {
			// we want to keep a map containing the true values of the vectors, in order to find the median
			std::vector<std::vector<int>> centroids_map(k);
			// insert the vectors in the map
			for (int i = 0; i < n_points; i++) {
				int assigned = assigned_centroid.at(i);
				centroids_map.at(assigned).push_back(i);
			}	
			return centroids_map;
		}
		
		// initialize our clusters using the init++ method
		void initialization(void) {
		// chose a first index randomly
		std::random_device mch;
		std::default_random_engine generator(mch());
		std::uniform_int_distribution<int> distribution_int(0,n_points - 1);
		int index = distribution_int(generator);
		// this is our first centroid
		centroids.push_back(feature_vectors.at(index));
			// find the other centroids with the algorithm init++
			for (int t = 1; t < k; t++) {
				// define a distance vector
				vector<T> D(n_points, INT_MAX);
				// define a vector of the partial sums
				vector<T> P(n_points, 0);
				// for every point
				for (int i = 0; i < n_points; i++) {
					// check if it is centroid
					vector<T> curr = feature_vectors.at(i);
					if (std::find(centroids.begin(), centroids.end(), curr) == centroids.end()) {
						//compute D(i)
						for (int j = 0; j < t; j++) {
							T dist = metrics::ManhatanDistance(feature_vectors.at(i), centroids.at(j), space_dim);
							if (dist < D.at(i))
								D.at(i) = dist;
						}
					} else {
						// if it is a centroid, set its D value to 0
						D.at(i) = 0;
					}
					// compute the partial sums effectively
					if (i == 0)
						P.at(i) = pow(D.at(0), 2);
					else 
						P.at(i) = P.at(i - 1) + pow(D.at(i), 2);
				}
				//pick a uniformly distributed x
				std::uniform_real_distribution<double> distribution_real(0, P.at(n_points -1));
				double x = distribution_real(generator);

				// find its correct position
				int r_index = -1;
				int i = 0;
				for (i = 0; i < n_points; i++) {
					if (P.at(i) > (T)x) {
						r_index = i;
						break;
					}
				}
				// ensure that the point is not a csntroid
				while ((i - 1 > 0) && P.at(i) == P.at(i - 1)) {
					r_index--;
				}
				assert(r_index > 0);
				// now the r_index has a correct value, so its our new centroid
				centroids.push_back(feature_vectors.at(r_index));
			}
		}

		// update the centroids with the kmedian method
		void update(void) {
			// we want to keep a map containing the true values of the vectors, in order to find the median
			std::vector<std::vector<std::vector<T>>> centroids_map(k);
			// insert the vectors in the map
			for (int i = 0; i < n_points; i++) {
				int assigned = assigned_centroid.at(i);
				centroids_map.at(assigned).push_back(feature_vectors.at(i));
			}
			// for each centroid 
			for (int i = 0; i < k; i++) {
				// for each one of its dimenions
				for (int j = 0; j < (int)space_dim; j++) {
					vector<T> current_component;
					// parse through all of its assigned vectors in order to find the median
					for (int n = 0; n < (int)centroids_map.at(i).size(); n++) {
						current_component.push_back(centroids_map.at(i).at(n).at(j));
					}
					centroids.at(i).at(j) = our_math::median(current_component);
				}
			}
		};

		// update the centroids with the kmedian method
		void update_original(void) {
			// we want to keep a map containing the true values of the vectors, in order to find the median
			std::vector<std::vector<std::vector<T>>> centroids_map(k);
			// insert the vectors in the map
			for (int i = 0; i < n_points; i++) {
				int assigned = assigned_centroid.at(i);
				centroids_map.at(assigned).push_back(original_feature_vectors.at(i));
			}
			// for each centroid 
			for (int i = 0; i < k; i++) {
				// for each one of its dimenions
				for (int j = 0; j < (int)original_space_dim; j++) {
					vector<T> current_component;
					// parse through all of its assigned vectors in order to find the median
					for (int n = 0; n < (int)centroids_map.at(i).size(); n++) {
						current_component.push_back(centroids_map.at(i).at(n).at(j));
					}
					original_centroids.at(i).at(j) = our_math::median(current_component);
				}
			}
		};

		// assign vectors to the nearest cluster using the lloyds algorithm
		int assignment_lloyds(void) {
			// keep track of the changes in assignments
			int changes = 0;
			// itterate each vector of the dataset
			for (int i = 0; i < n_points; i++) {
				int prev = assigned_centroid.at(i);
				assigned_centroid.at(i) = nearest_centroid(feature_vectors.at(i));
				// check and record changed centroids
				if (prev != assigned_centroid.at(i))
					changes++;
			}

			return changes;
		};
		int reverse_assignment(void) {
			// keep a vector of the new assignments
			vector<int> new_assigned(n_points, -1);
			// initial radius
			T radius = min_dist_between_centroids() / 2;
			// keep track of the changes
			int changes = 0;
			// set a threshold in order to break the loop
			int prev_unassigned = INT_MAX;
			// keep track of unassinged points
			int unassinged = INT_MAX - 1;
			// break the loop when all the balls contain no new vectors
			while (unassinged != prev_unassigned) {
				// do a range search query for every centroid
				for (int i = 0; i < k; i++) {
					list<pair<int, T>> result;
					// the type of range search depends on what the user wants
					if (assignment_method == "reverse_LSH")
						result = lsh_instant->RangeSearch(centroids.at(i), radius, 1);
					else if (assignment_method == "reverse_Hypercube")
						result = hc_instant->RangeSearch(centroids.at(i), radius, 1);

					// iterate all the results
					typename std::list<std::pair<int,T>>::iterator pair_it;
					for (pair_it = result.begin(); pair_it !=result.end(); pair_it++) {
						// get the current vector
						int curr_vec = pair_it->first;

						// if it is still unassigned
						if (new_assigned.at(curr_vec) == -1) {
							// temporarly assign it to this centroid
							new_assigned.at(curr_vec) = i;
						}
						// if it has been already assigned
						else {
							// chcek if its distance from the current centroid, is less than the previous' one
							int prev_assigned = new_assigned.at(curr_vec); 
							T prev_dist = metrics::ManhatanDistance(feature_vectors.at(curr_vec), centroids.at(prev_assigned), space_dim);
							T new_dist = pair_it->second;
							// if it is, it is closest to the current centroid, thus change the asssigned value in the temp vector
							if (new_dist < prev_dist)
								new_assigned.at(curr_vec) = i;
						}
					}
				}

				// update the unassigned vectors count
				prev_unassigned = unassinged;
				unassinged = count_unassigned(new_assigned);
				// update the radius
				radius *= 2;
			}

			// update the untracked vectors, and check for new changes
			for (int i = 0; i < n_points; i++) {
				// for each one not tracked, use direct assignment
				if (new_assigned.at(i) == UNASSIGNED)
					new_assigned.at(i) = nearest_centroid(feature_vectors.at(i));
				// check for changes
				if (assigned_centroid.at(i) != new_assigned.at(i))
					changes++;
			}

			// update the assigned vector
			assigned_centroid = new_assigned;
			return changes;
		};

		// compute the silhouette of each cluster, and the total one
		std::pair<std::vector<double>, double> compute_silhouette(void) {

			// create a temporary map to store the vectors that belong to each centroid
			std::vector<std::list<int>> centroids_map(k);

			for (int i = 0; i < n_points; i++) {
				centroids_map.at(assigned_centroid.at(i)).push_back(i);
			}

			// declare a vector of each cluster's silhouette
			vector<double> silhouettes(k, 0);

			// declare a vector that holds how many vectors are assinged to each centroid
			vector<int> assigned(k, 0);
			// declare the total silhouette
			double total_sil = 0;

			// compute the silhouette of each point
			for (int i = 0; i < n_points; i++) {
				// find the nearest and the second nearest 
				int nearest = assigned_centroid.at(i);
				int second_nearest = second_nearest_centroid(feature_vectors.at(i));
				assigned.at(nearest)++;
				double a_i = 0, b_i = 0;
				// compute a and b values for this vector
				for (std::list<int>::iterator it = centroids_map.at(nearest).begin(); it != centroids_map.at(nearest).end(); it++) {
					a_i += (double)metrics::ManhatanDistance(feature_vectors.at(i), feature_vectors.at(*it), space_dim);
				}	

				for (std::list<int>::iterator it = centroids_map.at(second_nearest).begin(); it != centroids_map.at(second_nearest).end(); it++) {
					b_i += (double)metrics::ManhatanDistance(feature_vectors.at(i), feature_vectors.at(*it), space_dim);
				}	

				// compute the vecor's silhouette
				double sil = (b_i - a_i) / std::max(a_i, b_i);

				// add it to the cluster's silhouette
				silhouettes.at(nearest) += sil;

				// add it to the total silhouette
				total_sil += sil;
			}

			// compute the final value of the silhouette for each cluster
			for (int i = 0; i < k; i++) {
				silhouettes.at(i) /= assigned.at(i);
			}

			// compute the final value of the total silhouette
			total_sil /= n_points;

			// return a pair of the vector of the silhouettes for each cluster and the total silhouette
			return make_pair(silhouettes, total_sil);
		}


		// run the clustering algorithm
		void run_clustering(void) {
			// Step 1: Initialize the centroids using init++
			initialization();

			// set manually a threshold for stoping the algorithm
			int threshold = n_points / 1000;
			// number of changed vector. It is going to change in each itteration
			int changed = INT_MAX;

			// repeat until the threshold is not reached
			while (changed > threshold) {
				// Step 2: call the appropriate method of assignment
				if (assignment_method == "lloyds")
					changed = assignment_lloyds();
				else 
					changed = reverse_assignment();
			
				// Step 3: Update the centroids
				update();
			}
			// if we are using the reduced vectors, we want to make the rest of the computations in the original dimension
			if (using_reduced_space) {
				// create the centroids from the info that we already have
				for (int i = 0; i < k; i++) {
					vector<T> temp_centroid(original_space_dim, 0);
					original_centroids.push_back(temp_centroid);
				}
				// update the centroids to get the median
				update_original();
				// no need for the reduced no more!
				centroids = original_centroids;
				feature_vectors = original_feature_vectors;
				space_dim = original_space_dim;
			}
		}

		// compute the objective function for evaluating the clusters
		double compute_objective_function() {
			double obj_sum = 0;
			// sum of the distanced of each image from their centroid
			for (int i = 0; i < n_points; i++) {
				obj_sum += metrics::ManhatanDistance(feature_vectors[i], feature_vectors[assigned_centroid[i]], space_dim);
			}
			return obj_sum;
		}

};