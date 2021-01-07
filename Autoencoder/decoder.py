from keras import layers, optimizers, losses, metrics
from keras.models import Sequential
from keras import Model
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dense
from keras import Input

def decoder(encoder_result):
    	# gather the info given by the encoder tuple
    prev_conv_layer, decoding_layers, conv_filter_size, current_filters_per_layer, shape = encoder_result

	# divide the filters by 2
    current_filters_per_layer /= 2

    # the decoding layers are one less than the encoding ones
    decoding_layers -= 1

    # find out how many neurons we want to convert to with a fc layer
    dense_neurons = shape[1] * shape[2] * shape[3]
    prev_conv_layer = Dense(dense_neurons)(prev_conv_layer)

    # reshape to what the image was before the flattening
    prev_conv_layer = Reshape(shape[1:])(prev_conv_layer)

    # for the first n-1 layers of the decoder
    for i in range (0, decoding_layers - 1):
        name1 = 'dec' + str(i) + 'a'
        name2 = 'dec' + str(i) + 'b'
        # apply convolution and batch normalization 2 times
        conv_layer = Conv2DTranspose(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name=name1)(prev_conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Conv2DTranspose(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name=name2)(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        
        # again, devide the filters per layer by 2
        current_filters_per_layer /= 2

        # to satisfy the next loop, the current layer becomes the previous one
        prev_conv_layer = conv_layer

    # after the completion of the loop, apply an upsampling technique
    upsampling = UpSampling2D((2,2), name='dec_last_1')(prev_conv_layer)


    # the last layer takes its input from the upsampling that we've performed
    last_conv_layer = Conv2DTranspose(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name='dec_last_2')(upsampling)
    last_conv_layer = BatchNormalization()(last_conv_layer)
    last_conv_layer = Conv2DTranspose(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same', name='dec_last_3')(last_conv_layer)
    last_conv_layer = BatchNormalization()(last_conv_layer)

    # apply one last time the upsampling technique
    upsampling = UpSampling2D((2,2), name='dec_last_4')(last_conv_layer)

    # the decoded array is produced by applying 2d convolution one last time, this one with a sigmoid activation function
    decoded = Conv2DTranspose(1, (conv_filter_size, conv_filter_size), name='dec_last_5', activation='sigmoid', padding='same')(upsampling)

    return decoded