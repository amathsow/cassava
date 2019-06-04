from keras.layers import (Dense, Input, Activation, Flatten, Conv2D,
                          MaxPooling2D, GlobalAveragePooling2D, add)
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


def residualMapping(inputTensor, filters):
    """
    Residual building block where input and output tensor are the same
    dimensions
    """
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = Activation('relu')(x)

    x = add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def downsizeMapping(inputTensor, filters):
    """
    Residual building block where input tensor dimensions are halved, but
    feature map dimensions double
    """
    x = Conv2D(filters, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
    x = Activation('relu')(x)

    inputTensor = Conv2D(filters, (1, 1), strides=(2, 2), kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def ResNet34Bottom(inputShape):
    """
    Creates a stack of layers equivalent to ResNet-34 architecture up until the
    average pool and 1000-d fully connected layer.
    Assumes that the input is a square patch, so that the output is 1x1x512
    regardless of input dimensions (using average pooling)
    """
    inputTensor = Input(inputShape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(inputTensor)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residualMapping(x, 64)
    x = residualMapping(x, 64)

    x = Conv2D(128, (1,1), strides=(2,2), padding='same', activation='relu')(x)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128) 

    x = Conv2D(256, (2,2), strides=(2,2), padding='same', activation='relu')(x)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)

    x = Conv2D(512, (1,1), padding='same', activation='relu')(x)
    x = residualMapping(x, 512)
    x = residualMapping(x, 512)

    model = Model(inputTensor, x, name='ResNet34Bottom')
    return model
