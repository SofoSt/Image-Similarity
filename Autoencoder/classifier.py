	
import argparse
import sys

sys.path.append('..')
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


import webbrowser


import hiplot as hip

from common.mnist_parser import *
from common.utils import *

from encoder import *
from decoder import *

# Build the model, consisti
def fully_connected(endoder, n_neurons):
    flat = Flatten()(endoder)
    dense = Dense(n_neurons, activation='relu')(flat)
    # dropout = Dropout(0.5, input_shape=(None, n_neurons))(dense);
    output = Dense(10, activation='softmax')(dense)

    return output


def main():

    # create a parser in order to obtain the arguments
    parser = argparse.ArgumentParser(description='Create a an image classifier NN using a pre-trained encoder')
    # the parse all the arguments needed from the program
    parser.add_argument('-d', '--trainset', action='store', default=None,  metavar='', help='Relative path to the training set')
    parser.add_argument('-dl', '--trainlabels', action='store', default=None,  metavar='', help='Relative path to the training labels')
    parser.add_argument('-t', '--testset', action='store', default=None,  metavar='', help='Relative path to the test set')
    parser.add_argument('-tl', '--testlabels', action='store', default=None,  metavar='', help='Relative path to the test labels')
    parser.add_argument('-model', '--model', action='store', default=None,  metavar='', help='Relative path to the h5 file of the trained encoder model')

    # parse the arguments
    args = parser.parse_args()

    print("------PARSING THE DATA.....------")
    # parse the train and test datasets
    train_X_full, rows, columns = parse_X(args.trainset)
    train_Y_full = parse_Y(args.trainlabels)
    
    test_X, _, _ = parse_X(args.testset)
    test_Y = parse_Y(args.testlabels)

    # convert the labels into categorical arrays, in order to classify
    new_train_Y_full = to_categorical(train_Y_full)
    new_test_Y = to_categorical(test_Y)

    # normalize all values between 0 and 1 
    train_X_full = train_X_full.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.

    # reshape the train and validation matrices into 28x28x1, due to an idiomorphy of the keras convolution.
    train_X_full = np.reshape(train_X_full, (len(train_X_full), rows, columns, 1))
    test_X = np.reshape(test_X, (len(test_X), rows, columns, 1))

    # split the train set in order to have a set for validation of our classifier
    train_X ,valid_X, train_ground, valid_ground = train_test_split(train_X_full, new_train_Y_full, test_size=0.2, random_state=42)

    # load the autoencoder that was given as an argument
    loaded_encoder = load_model(args.model)

    layers = loaded_encoder.layers
    for idx, layer in enumerate(layers):
        if (layer.name == 'enc_dropout'):
            break

    input_img = Input(shape = (rows, columns, 1))
    # create an empty list in order to store the models
    models_list = []
    plots = 0

    while 1:
            choice = int(input("Choose one of the following options to procced: \n \
                        1 - Chose hyperparameters for the next model's training\n \
                        2 - Print the plots gathered from the expirement\n \
                        3 - Predict the test set using one of the pretrained models and exit\n \
                        4 - Load a pre-trained classifier\n"))
            if (choice == 1):
                # hyperparameters input
                epochs = int(input("Give the number of epochs\n"))
                batch_size = int(input("Give the batch size\n"))
                fc_n_neurons = int(input("Give the number of neurons in the fully connected layer\n"))

                # create the model concisting of the encoder and a fully connected layer
                new_model = Model(loaded_encoder.inputs,loaded_encoder.layers[idx].output)
                # new_model.summary()

                encode = new_model(input_img)
                full_model = Model(input_img, fully_connected(encode, fc_n_neurons))
                """
                In order to achieve better speed performance, we are going to train in 2 steps:
                
                    - 1st step: train only the FC layer
                    - 2nd step: given the results of the 1st step, train the whole model
                """
                # mark every other layer except of the dense as untrainable
                for layer in full_model.layers:
                    if ("dense" not in layer.name):
                        layer.trainable = False

                # compile the model
                full_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
                # and train it
                classify_train = full_model.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

                # in the 2nd step, every layer is trainable
                for layer in full_model.layers:
                    layer.trainable = True

                # compile and train the49 model
                full_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
                
                classify_train = full_model.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

                # create a dictionary of the model plus some info in order to save it to the models' list
                model_plus_info = {'full_model': full_model, 'epochs': epochs, 'batch_size': batch_size, 'fully_connected_neurons': fc_n_neurons}
                models_list.append(model_plus_info)
        
                # create a file name and save the file
                name = "../Models/Classifier/" + str(epochs) + "_" + str(batch_size) + "_" + str(fc_n_neurons) + ".h5"

            # Print stats for the last model and the models in general
            elif (choice == 2):
                # Get the last model from the list
                last_model = models_list[-1]
                model = last_model["full_model"]
                epochs = last_model["epochs"]

                # create a subplot to plot the accuracy and the loss

                accuracy = model.history.history['accuracy']
                if (len(accuracy) > 1):	
                    print("\n--------------- LAST TRAINED MODEL METRICS ----------------\n")	
                    # plot the validation and the train accuracy and errors
                    val_accuracy = model.history.history['val_accuracy']
                    loss = model.history.history['loss']
                    val_loss = model.history.history['val_loss']
                    epochs = range(len(accuracy))

                    plt.plot(epochs,accuracy, label = 'training accuracy')
                    plt.plot(epochs, val_accuracy, label = 'validation accuracy')
                    plt.legend()
                    plt.ylim([0.9, 1])

                    plt.show()
            
                    plt.plot(epochs,loss,  label = 'training loss')
                    plt.plot(epochs,val_loss,  label = 'validation loss')
                    plt.legend()
                    plt.ylim([0, 0.2])
            
                    plt.show()
                else: 
                    print("Last model was loaded. Not enough data to plot")

                if (len(models_list) > 1):
                    print("\n--------------- ALL TRAINED MODElS METRICS-----------------\n")	
                    """
                    Using hiPlot to create a high dimensional plot of our models' hyperparameters
                    """
                    plotting_dict_list = []
                    for m in models_list:
                        m["loss"] = round(m['full_model'].history.history['loss'][-1], 4)
                        m["accuracy"] = round(m['full_model'].history.history['accuracy'][-1], 4)
                        plotting_dict_list.append(dict(list(m.items())[1:]))

                        # get the result of hiplot in an html page
                        try:
                            html_str = hip.Experiment.from_iterable(plotting_dict_list).to_html(force_full_width=True)
                            # open a file to save the html containing the plot 
                            f_name = "../Results/hiplots/hiplot_result_" + str(plots) + '.html'
                            # increase the plot variable
                            plots += 1
                            f = open(f_name, "w")
                            f.write(html_str)
                            f.close()
                            # pop-up in google chrome
                            webbrowser.get('/usr/bin/google-chrome %s').open(f_name)
                        except:
                            print("HiPlot not installed, hi-dimensional plot can not be created")
                else:
                    print("There is only one model, train an other one too to compare...\n")

            # Predict the data and visualize it
            elif (choice == 3):
                print("Select the trained model that you want to use")
                i = 0
                # give the available models to the user as options
                for model in models_list:
                    print(i, "- epochs: ", model['epochs'], ", batchsize: ", model['batch_size'], " n. neurons: ", model['fully_connected_neurons'], "\n")
                    i += 1

                # get the selection and load the appropriate model
                selection = int(input())

                model = models_list[selection]["full_model"]

                clust_dict = {new_list: [] for new_list in range(10)}

                predicted_classes = model.predict(train_X_full)
                predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

                for i, label in enumerate(predicted_classes):
                    clust_dict[label].append(i)

                f = open("classes_for_clustering", "w")
                for i in range(len(clust_dict)):
                    f.write("CLUSTER-" + str(i) + " { size: " + str(len(clust_dict[i])) + ", ")
                    for n in clust_dict[i]:
                        f.write(str(n) + ", ")
                
                    f.write("}\n")    
                
                f.close()
                # exit the program
                break;
            elif (choice == 4):
                # Get the model info
                path = input("Give the path and the name of the model you want to load\n")
                epochs = int(input("Give the number of epochs that were used to train the model\n"))
                batch_size = int(input("Give the batch size that was used to train the model\n"))
                fc_n_neurons = int(input("Give the number of neurons that were used to train the model\n"))
                # load the pre-trained model
                loaded_model = load_model(path)
                loaded_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
                
                classify_train = loaded_model.fit(train_X, train_ground, batch_size=batch_size, epochs=1, verbose=1, validation_data=(valid_X, valid_ground))
                # collect the info in the dictionary
                model_plus_info = {'full_model': loaded_model, 'epochs': epochs, 'batch_size': batch_size, 'fully_connected_neurons': fc_n_neurons}
                # append the model in the models' list
                models_list.append(model_plus_info)
            else:
                print("Choose one of the default values")

if __name__ == '__main__':
    main()