import os
import struct

import numpy as np

# convert a number from big to little endian. Needed for the parsing of MNIST
def read_big_endian(byte_sequence):
	return int.from_bytes(byte_sequence, "big")

# Parse the images of the MNIST dataset
def parse_X(path, n_requested):
	# open the file given as an argument
	with open(path, "rb") as file:
		# read the magic number and catch a possible exception
		magic_number = read_big_endian(file.read(4))
		if magic_number < 0:
			raise Exception("Wrong magic number!")

		# read specifics about the images and their dimensions
		n_of_images = read_big_endian(file.read(4))
		rows = read_big_endian(file.read(4))
		columns = read_big_endian(file.read(4))
		
		if n_requested != n_of_images:
			n_of_images = n_requested
		# we are going to store our result in an np array
		result = []

		# for each image start filling the vectors
		for i in range(n_of_images):
    		# create a vector to store our image data
			lst = []
			# store each byte of the image in the vector, by parsing boh of the image's dimensions 
			for j in range(rows):
				for k in range(columns):
    				# read the pixel and convert it to little endian
					number = read_big_endian(file.read(1))
					# apppend it to the image's vector
					lst.append(number)
			# append the flatten vector of the image in the final one
			vec = np.array(lst)
			result.append(vec)
	# return the vector containig all the images, as well as the rows and the columns of each image
	return np.array(result), rows, columns

# parse the labels of the MNIST dataset
def parse_Y(path, n_requested):
	# open the labels file
	with open(path, "rb") as file:

		# read the magic number and catch a possible exception
		magic_number = read_big_endian(file.read(4))
		if magic_number < 0:
			raise Exception("Wrong magic number!")

		# read the number of lablels
		n_of_labels = read_big_endian(file.read(4))
		if n_requested != n_of_labels:
			n_of_labels = n_requested
		# we are going to store our result in a 1d np array
		result = np.zeros((n_of_labels))
		current_label = 0

		# parse through all the labels
		while current_label < n_of_labels:
			# and store each one in the correct position of the array
			number = read_big_endian(file.read(1))
			result[current_label] = int(number)
			current_label += 1

	return result
