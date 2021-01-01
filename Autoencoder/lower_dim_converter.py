import argparse
import sys
import struct

import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras import Model


from keras.optimizers import RMSprop
from keras.optimizers import Adam

from keras.models import Sequential
from keras.models import load_model


from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras import Input

from keras.backend import flatten

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from common.mnist_parser import *
from common.utils import *

def main():

    # create a parser in order to obtain the arguments
    parser = argparse.ArgumentParser(description='Create a an image classifier NN using a pre-trained encoder')

    # the parse all the arguments needed from the program
    parser.add_argument('-d', '--trainset', action='store', default=None,  metavar='', help='Relative path to the training set')
    parser.add_argument('-dl', '--trainlabels', action='store', default=None,  metavar='', help='Relative path to the training labels')
    parser.add_argument('-t', '--testset', action='store', default=None,  metavar='', help='Relative path to the test set')
    parser.add_argument('-tl', '--testlabels', action='store', default=None,  metavar='', help='Relative path to the test labels')
    parser.add_argument('-m', '--model', action='store', default=None,  metavar='', help='Relative path to the h5 file of the trained encoder model')

    # parse the arguments
    args = parser.parse_args()

    print("------PARSING THE DATA.....------")
    # parse the train and test datasets
    train_X, rows, columns = parse_X(args.trainset)
    train_Y = parse_Y(args.trainlabels)
    test_X, _, _ = parse_X(args.testset)
    test_Y = parse_Y(args.testlabels)

    # convert the labels into categorical arrays, in order to classify
    new_train_Y = to_categorical(train_Y)
    new_test_Y = to_categorical(test_Y)

    # normalize all values between 0 and 1 
    train_X = train_X.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.

    # reshape the train and validation matrices into 28x28x1, due to an idiomorphy of the keras convolution.
    train_X = np.reshape(train_X, (len(train_X), rows, columns, 1))
    test_X = np.reshape(test_X, (len(test_X), rows, columns, 1))

    # load the autoencoder that was given as an argument
    loaded_encoder = load_model(args.model)

    # hold all the layers until the latent one
    layers = loaded_encoder.layers
    for idx, layer in enumerate(layers):
        if (layer.name == 'enc_latent'):
                break

    input_img = Input(shape = (rows, columns, 1))

    # create the model concisting of the encoder
    new_model = Model(loaded_encoder.inputs,loaded_encoder.layers[idx].output)

    # create and compile the model that we will use in order to reduce the dimensions of the image
    encode = new_model(input_img)
    full_model = Model(input_img, encode)

    # compile the model
    full_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

    # get some info about the dimensions and the datasets' sizes
    n_train_images = train_X.shape[0]

    # get the list of the reduced images, and flatten it
    transformed_train = full_model.predict(train_X)
    transformed_train = transformed_train.flatten()

    # find out the dimension that the user has specified in the encoding model
    dimension = transformed_train.size // n_train_images
    # keep those for later normalization
    max_n_train = max(transformed_train)
    upper = 2**16 - 1

    # open a new binary file, and add...
    new_train_file = open("misc/train_reduced", "wb")
    # magic number
    new_train_file.write(struct.pack('>I', 1234))
    # number of images
    new_train_file.write(struct.pack('>I', n_train_images))
    # number of rows
    new_train_file.write(struct.pack('>I', dimension))
    # number of columns (always equal to 1)
    new_train_file.write(struct.pack('>I', 1))

    # normalize the data, in order for the pixels to be short ints
    normalized_new_train = [int(upper * x / max_n_train) for x in transformed_train]

    # write every pixel of every image in the new file
    for pixel in normalized_new_train:
        new_train_file.write(struct.pack('>H', int(pixel)))

    new_train_file.close()

    # same procedure for the test dataset
    n_test_images = test_X.shape[0]


    transformed_test = full_model.predict(test_X)

    transformed_test = transformed_test.flatten()

    max_n_test = max(transformed_test)

    new_test_file = open("misc/test_reduced", "wb")
    # magic number
    new_test_file.write(struct.pack('>I', 1234))
    # number of images
    new_test_file.write(struct.pack('>I', n_test_images))
    # number of rows
    new_test_file.write(struct.pack('>I', dimension))
    # number of columns (always equal to 1)
    new_test_file.write(struct.pack('>I', 1))

    normalized_new_test = [int(upper * x / max_n_test) for x in transformed_test]

    for pixel in normalized_new_test:
        new_test_file.write(struct.pack('>H', int(pixel)))

    new_test_file.close()


if __name__ == '__main__':
    main()