from keras import layers, optimizers, losses, metrics
from keras.models import Sequential
from keras import Model
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dense
from keras import Input

# encoding function for the autoencoder
"""
FUNCTION USAGE: The user gives the desired hyperparameters as following:
 - input_image: the shape of the images that the model is going to predict later
 - conv_layers: the number of convolution layers that the encoder will apply to the image
 - conv_filter_size: The size of the filter during each one of the convolution layers
 - n_conv_filters_per_layer: The number of filters in the first layer, each time multiplied by 2

 The encoder applies convolution, and batch normalization as many times as the 
 user has requested, and applies pooling after the 2 first layers, and finally a latent layer,
 and returns a tuple containing all the usefull info (result + hyperparameters).
"""
def encoder(input_image, conv_layers, conv_filter_size, n_conv_filters_per_layer, latent_dim = 10):
    
    # first layer: differs because it takes as input the shape of the image
    first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name='enc0a')(input_image)
    first_layer = BatchNormalization()(first_layer)
    first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name='enc0b')(first_layer)
    first_layer = BatchNormalization()(first_layer)

    # pooling after first layer
    pool = MaxPooling2D(pool_size=(2,2))(first_layer)
    dropout = Dropout(0.3, name="enc_dropout1")(pool)

    # the encoding layers are the number of layers given to us as an argument (TODO: possibly change)
    encoding_layers = conv_layers

    # each time, we will be nultiplying the filters per layer by 2
    current_filters_per_layer = 2 * n_conv_filters_per_layer

    # for the remaining encoding layers
    for i in range (1, encoding_layers):
        name1 = 'enc' + str(i) + 'a'
        name2 = 'enc' + str(i) + 'b'
        name3 = 'enc' + str(i) + 'c'
        # the first 2 take an input from the pooling
        if (i == 1 or i == 2):
            conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), name=name1, activation='relu', padding='same')(dropout)
        # the others from the previous convolution layer
        else:
            conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), name=name1, activation='relu', padding='same')(conv_layer)

        # after that, wy apply batch normalization, convolution and the again batch normalization
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), name=name2, activation='relu', padding='same')(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        # on the 1st layer in the loop(aka the 2nd), we want pooling
        if (i == 1):
            pool = MaxPooling2D(pool_size=(2,2), name=name3)(conv_layer)
            dropout = Dropout(0.5, name="enc_dropout2")(pool)
        # each time (except the last one, we multiply the filters per layer by 2)
        if (i < encoding_layers - 1):
            current_filters_per_layer *= 2
    # dropout layer, while holding the shape of the result for further use
    dropout = Dropout(0.3, name="enc_dropout")(conv_layer)
    shape = dropout.shape
    flat = Flatten(name="enc_flatten")(dropout)

    # latent layer
    latent = Dense(latent_dim, activation="relu", name="enc_latent")(flat)

    # return a tuple containing all the usefull info that we gathered from the encoder
    return (latent, encoding_layers, conv_filter_size, current_filters_per_layer, shape)

