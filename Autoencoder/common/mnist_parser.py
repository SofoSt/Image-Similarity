import os
import struct

import numpy as np

# convert a number from big to little endian. Needed for the parsing of MNIST
def read_big_endian(byte_sequence):
	return int.from_bytes(byte_sequence, "big")

# Parse the images of the MNIST dataset
def parse_X(path):
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
		
		# we are going to store our result in an np array
		result = np.zeros((n_of_images, rows, columns))
		curr_row = curr_col = curr_image = 0

		# parse all the images
		while (curr_image < n_of_images):
			# for every column and row, read each byte
			number = read_big_endian(file.read(1))
	 		
			# and correctly assign it to the result array
			result[curr_image, curr_row, curr_col] = number

			# change those variables while parsing
			if (curr_row == rows - 1 and curr_col == columns - 1):
				curr_image += 1
				curr_row = 0
				curr_col = 0
			elif (curr_col == columns - 1):
				curr_row += 1
				curr_col = 0
			else:
				curr_col += 1
	
	return result, rows, columns


# parse the labels of the MNIST dataset
def parse_Y(path):
	# open the labels file
	with open(path, "rb") as file:

		# read the magic number and catch a possible exception
		magic_number = read_big_endian(file.read(4))
		if magic_number < 0:
			raise Exception("Wrong magic number!")

		# read the number of lablels
		n_of_labels = read_big_endian(file.read(4))

		# we are going to store our result in a 1d np array
		result = np.zeros((n_of_labels))
		current_label = 0

		# parse through all the labels
		while current_label < n_of_labels:
			# and store each one in the correct position of the array
			number = read_big_endian(file.read(1))
			result[current_label] = number
			current_label += 1

	return result

