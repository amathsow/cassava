from keras.models import Model
from keras.layers import (Input, Conv2D, Dense, MaxPool2D, Dropout, Activation,
                          Concatenate, Flatten, GlobalMaxPooling2D,
                          Subtract, Add)
from keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet169
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from ResNet34Bottom import ResNet34Bottom
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import h5py


class EgoCNN:
    """
    Implementation of the egomotion neural network for pretext learning

    Arguments:
        in_size -- tensor shape of a single image
        out_size -- tensor shape of the output(s), should be coherent
        with tbins and xrange, yrange
        handle -- hdf5 dataset handle for the batch_generator method
        xrange -- (min,max) tuple with the extremes of x random
        ranslations (in pixels)
        yrange -- (min,max) tuple with the extremes of y random
        translations (in pixels)
        lscale -- multiplier for random translations
        trange -- (min,max) tuple with random rotation extremes
        tbins -- number of bins to digitize the rotations
    """
    def __init__(self, in_size, out_size,
                 batch_size=10,
                 handle=None,
                 xrange=(-3, 3), yrange=(-3, 3), lscale=7,
                 trange=(-30, 30), tbins=20):
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        # +1 because np.random.randint samples from [min, max+1]
        self.xrange = (xrange[0], xrange[1]+1)
        # +1 because np.random.randint samples from [min, max+1]
        self.yrange = (yrange[0], yrange[1]+1)
        self.lscale = lscale
        self.trange = trange
        self.tbins = tbins
        self.model = self.ego_cnn()

    def bcnn(self):
        '''
        Siamese CNN which makes up the bottom part of the network together
        with its twin.
        '''
        input_tensor = Input(shape=self.in_size)
        x = Conv2D(96, kernel_size=(7, 7), strides=(1, 1), padding='same',
                   activation='relu')(input_tensor)
        x = MaxPool2D(pool_size=(3, 3))(x)

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3))(x)

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), name='bcnn_output')(x)

        x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
        return Model(input_tensor, x)

    def tcnn(self, bottom_shape=None):
        '''
        Conv2D and Fully connected layers to be plugged on top of the concat
        siamese CNNs.
        Arguments:
            bottom_shape -- output tensor of the concat bcnn's
        '''
        if bottom_shape is None:
            raise Exception('bottom_shape not specified (None)')
        input_streams = Input(shape=bottom_shape)
        x = Conv2D(64, kernel_size=(1, 1), activation='relu')(input_streams)
        x = Flatten()(x)
        x = Dense(300, activation='elu')(x)
        Y = Dropout(rate=.3)(x)
        return Model(input_streams, Y)

    def ego_cnn(self):
        '''
        Keras model putting together bottom and top NN
        '''
        in_stream1 = Input(shape=self.in_size, name='input0')
        in_stream2 = Input(shape=self.in_size, name='input1')
        bottom = self.bcnn()
        middle = Concatenate()([bottom(in_stream1), bottom(in_stream2)])
        middle_size = tuple(a*b
                            for a, b
                            in zip(bottom.output_shape[1:], [1, 1, 2]))
        top = self.tcnn(middle_size)
        output = top(middle)
        output = [Dense(size,
                        activation='softmax',
                        kernel_regularizer=l2(0.001),
                        bias_regularizer=l2(0.001),
                        name=f'output{num}')(output)
                  for num, size
                  in enumerate(self.out_size)]
        return Model([in_stream1, in_stream2], output)

    def batch_generator(self, handle):
        """
        Extract a random batch of images, apply transformations and return
        both, together with the corresponding transformation parameters
        Arguments:
            handle -- hdf5 dataset handle
        """
        # extract batch_size random indices from the dataset for SGD
        total_size = handle.shape[0]
        while True:
            indices = np.random.permutation(
                        range(total_size))[0:self.batch_size]
            indices = np.sort(indices)
            # define batch_size x and y translations as integers in
            # xrange and yrange
            x_trans = np.random.randint(*self.xrange, size=self.batch_size)
            y_trans = np.random.randint(*self.yrange, size=self.batch_size)
            # define batch_size rotations in trange
            z_rotat = np.random.uniform(*self.trange, size=self.batch_size)
            # group them in tbins
            z_rotat = np.digitize(z_rotat,
                                  np.linspace(*self.trange, self.tbins+1,
                                              endpoint=True))
            # group each x, y , z transformation in a list of
            # batch_size dictionaries
            trans_list = [{'theta': t, 'tx': x*self.lscale,
                           'ty': y*self.lscale} for t, x, y in zip(z_rotat,
                                                                   x_trans,
                                                                   y_trans)]
            # apply transformations to a batch (hdf5 dataset accepts lists for
            # slicing, not np.arrays)
            datagen = ImageDataGenerator()
            Xbatch_trans = np.array([datagen.apply_transform(img, tran)
                                     for img, tran
                                     in zip(handle[list(indices)],
                                            trans_list)])
            Xbatch = handle[list(indices)]
            # turning classes into categorical values
            # classes need to be positive
            Ybatch_tx = to_categorical(x_trans-self.xrange[0],
                                       sum(np.absolute(self.xrange)))
            # classes need to be positive
            Ybatch_ty = to_categorical(y_trans-self.yrange[0],
                                       sum(np.absolute(self.yrange)))
            # classes need to start from 0
            Ybatch_rot = to_categorical(z_rotat-1, self.tbins)
            yield ({'input0': Xbatch, 'input1': Xbatch_trans},
                   {'output0': Ybatch_rot, 'output1': Ybatch_tx,
                    'output2': Ybatch_ty})


class EgoCNNReg(EgoCNN):
    """
    Casting the self-supervised problem into a regression task
    """
    def batch_generator(self, handle):
        """
        Extract a random batch of images, apply transformations
        and return both, together with the corresponding
        transformation parameters

        #Arguments
           handle: hdf5 dataset
        """
        # extract batch_size random indices from the dataset for SGD
        total_size = handle.shape[0]
        while True:
            indices = np.random.permutation(
                        range(total_size))[0:self.batch_size]
            indices = np.sort(indices)
            # define batch_size x and y translations as
            # integers in xrange and yrange
            x_trans = np.random.randint(*self.xrange, size=self.batch_size)
            y_trans = np.random.randint(*self.yrange, size=self.batch_size)
            # define batch_size rotations in trange
            z_rotat = np.random.uniform(*self.trange, size=self.batch_size)
            # group each x, y , z transformation in a list
            # of batch_size dictionaries
            trans_list = [{'theta': t,
                           'tx': x*self.lscale,
                           'ty': y*self.lscale} for t, x, y in zip(z_rotat,
                                                                   x_trans,
                                                                   y_trans)]
            # apply transformations to a batch
            # (hdf5 dataset accepts lists for slicing, not np.arrays)
            datagen = ImageDataGenerator()
            Xbatch_trans = np.array([datagen.apply_transform(img, tran)
                                     for img, tran
                                     in zip(handle[list(indices)],
                                            trans_list)])
            Xbatch = handle[list(indices)]
            yield ({'input0': Xbatch, 'input1': Xbatch_trans},
                   {'output0': z_rotat,
                    'output1': x_trans,
                    'output2': y_trans})


# Some handy subclasses

class EgoDense(EgoCNN):
    def bcnn(self):
        model = DenseNet169(include_top=False,
                            weights='imagenet',
                            input_shape=self.in_size,
                            pooling=None)
        return model


class EgoInception(EgoCNN):
    def bcnn(self):
        model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=self.in_size,
                            pooling=None)
        return model


class EgoMobile(EgoCNN):
    def bcnn(self):
        model = MobileNetV2(include_top=False,
                            weights='imagenet',
                            input_shape=self.in_size,
                            pooling=None)
        return model


class EgoMobileReg(EgoCNNReg):
    def bcnn(self):
        model = MobileNetV2(include_top=False,
                            weights='imagenet',
                            input_shape=self.in_size,
                            pooling=None)
        return model


class EgoResNet(EgoCNN):
    def bcnn(self):
        model = ResNet34Bottom(inputShape=self.in_size)
        return model
