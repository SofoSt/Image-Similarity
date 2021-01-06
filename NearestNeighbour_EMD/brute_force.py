import sys
"""
Breute FOrce class, that is being initialized by passsing all the 
vectors from the train dataset, and then runs a kNearest Neighbour
given the query vector.
"""
class BruteForce():
	# Initialization of the class
	def __init__(self, feature_vectors, metric, n_clusters):
		self.feature_vectors = feature_vectors
		self.metric = metric
		self.n_clusters = n_clusters

	def kNearestNeighbour(self, query, k):
		# list for the neighbours to be stored
		result = []
		kth_min_distance = sys.maxsize

		for i, feature in enumerate(self.feature_vectors):
			distance = self.metric(feature, query, self.n_clusters)

			if len(result) > 0 and len(result) < k:
				
				if distance >= kth_min_distance:
					kth_min_distance = distance
					result.append((i, distance))
				else:
					for j, (_, d) in enumerate(result):
						if (distance < d):
							result.insert(j, (i, distance))
							break
				
			elif len(result) > 0 and len(result) == k:
				if (distance < kth_min_distance):
					for j, (_, d) in enumerate(result):
						if (distance < d):
							result.insert(j, (i, distance))
							break
					result.pop(-1)
					_, kth_min_distance = result[-1]
		   
			elif len(result) == 0:
				kth_min_distance = distance
				result.append((i, distance))
			else:
				print("kNearest Problem")

		return result
