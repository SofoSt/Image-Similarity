import numpy as np
from brute_force import *
from parse_flat import *
from EMD import *
import argparse


# define the manhatan metric
def manhattan_distance(a, b, _):
	return np.abs(a - b).sum()

def main():
    	
	# create a parser in order to obtain the arguments
	parser = argparse.ArgumentParser(description='Predict the k Nearest Neighbours of an image')

	parser.add_argument('-d', '--dataset', action='store', default=None,  metavar='', help='Relative path to the train dataset')
	parser.add_argument('-q', '--queryset', action='store', default=None,  metavar='', help='Relative path to the test dataset')
	parser.add_argument('-l1', '--train_labels', action='store', default=None,  metavar='', help='Relative path to the train labels')
	parser.add_argument('-l2', '--test_labels', action='store', default=None,  metavar='', help='Relative path to the test labels')
	# parse the arguments
	args = parser.parse_args()

	# parse the dataset and the queryset, in order to create flat vectors
	features, _, _ = parse_X(args.dataset, 2000)
	queries, _, _ = parse_X(args.queryset, 200)

	# parse the labels
	feature_labels = parse_Y(args.train_labels, 2000)
	query_labels = parse_Y(args.test_labels, 200)

	# initialize the BF classes, by passing the feature vectors and the desired metric
	bf_class_manh = BruteForce(features, manhattan_distance, 4)
	bf_class_EMD = BruteForce(features, EMD, 4)

	# we want the 10 nearest neighbours, hardcoded
	n_neighs = 10

	# keep a counter of the correctly computed labels
	correctly_computed_manh = 0
	correctly_computed_EMD = 0

	# run all the queries
	for i, query in enumerate(queries):
		if (i == 10):
			break
		print(i)
    	# list of tuple of the NNs and their distances, using manhattan
		result = bf_class_manh.kNearestNeighbour(query, n_neighs)
		# itterate through the neigbours' list, and check how many correct labels were found
		for neighbour in result:
			if feature_labels[neighbour[0]] == query_labels[i]:
				correctly_computed_manh += 1

		# list of tuple of the NNs and their distances, using EMD
		result = bf_class_EMD.kNearestNeighbour(query, n_neighs)
		# itterate through the neigbours' list, and check how many correct labels were found
		for neighbour in result:
			if feature_labels[neighbour[0]] == query_labels[i]:
				correctly_computed_EMD += 1

	print("Average Correct Search Results EMD: ", (correctly_computed_EMD / n_neighs * 100)/ queries.shape[0], "%")
	print("Average Correct Search Results MANHATTAN: ", (correctly_computed_manh / n_neighs * 100)/ queries.shape[0], "%")

# Main function of the autoencoder
if __name__ == "__main__":
	main()
