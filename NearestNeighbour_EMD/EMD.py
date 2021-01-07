from pulp import *
import numpy as np
from matplotlib import pyplot as plt
import math

def create_clusters(array, nrows, ncols):

	r, h = array.shape
	return (array.reshape(h//nrows, nrows, -1, ncols)
				 .swapaxes(1, 2)
				 .reshape(-1, nrows, ncols))


class EMD():
	def __init__(self, n_clusters):
		# compute the distanced
		self.d = np.zeros((n_clusters, n_clusters))

		for i in range(n_clusters):
			for j in range(n_clusters):
				row_i = i / (math.sqrt(n_clusters))
				row_j = j / (math.sqrt(n_clusters))
				col_i = i % (math.sqrt(n_clusters))
				col_j = j % (math.sqrt(n_clusters))
				self.d[i][j] = math.sqrt((row_i - row_j) ** 2 + (col_i - col_j) ** 2)

	def compute_EMD(self, image_a, image_b, n_clusters):	
		# find out the dimenstion of the image
		dim = int(math.sqrt(len(image_a)))
		# in order to reshape them
		image_a = image_a.reshape(dim,-1)
		image_b = image_b.reshape(dim,-1)
		# find the dimension of the clusters
		cluster_dim = int(dim // math.sqrt(n_clusters))
		# split the images in order to create the clusters
		clusters_a = create_clusters(image_a, cluster_dim, cluster_dim)
		clusters_b = create_clusters(image_b, cluster_dim, cluster_dim)
		# we need the array of the distances between the representatives
		d = np.zeros((n_clusters, n_clusters))
		# and 2 arrays to hold the sum of the images
		w_a = np.zeros(n_clusters)
		w_b = np.zeros(n_clusters)
		
		# compute the sums
		for i in range(n_clusters):
			w_a[i] = np.sum(clusters_a[i])
			w_b[i] = np.sum(clusters_b[i])
		
		# normalize the w values
		w_a = w_a / np.sum(w_a)
		w_b = w_b / np.sum(w_b)

		# set the variables for the LP problem
		F = []
		for i in range(n_clusters):
			temp_f = []
			for j in range(n_clusters):
				temp_f.append(LpVariable("f" + str(i) + "_" + str(j), lowBound = 0))
			F.append(temp_f)

		# define the LP Problem
		emd_problem = LpProblem("EMD", LpMinimize)

		# define the objective function
		constraints = []
		obj_fn = []
		for i in range(n_clusters):
			for j in range(n_clusters):
				obj_fn.append(F[i][j] * self.d[i][j])

				constraints.append(F[i][j])
			
		# we want to minimize the sum
		emd_problem += lpSum(obj_fn)

		# 2nd and 3rd constraints
		for i in range(n_clusters):
			const_a = [F[i][j] for j in range(n_clusters)]
			emd_problem += lpSum(const_a) == w_a[i]

		for j in range(n_clusters):
			const_b = [F[i][j] for i in range(n_clusters)]
			emd_problem += lpSum(const_b) == w_b[j]

		# insert the 4rd constraint 
		# emd_problem += lpSum(constraints) == np.sum(w_a)
		
		# solve the problem
		emd_problem.writeLP("../misc/EMD.lp")
		emd_problem.solve(GLPK_CMD(msg=False))

		# return the minimum flow, but normalized
		return value(emd_problem.objective)