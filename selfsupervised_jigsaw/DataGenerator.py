import h5py
import random
import time
import numpy as np
#from image_preprocessing import image_transform
import itertools
import warnings
import threading

from PIL import Image
import sys




class JigsawCreator:
    """
    Creates an image processor that converts an image passed as a numpy array
    into 9 subimages, applies processing to them to improve the generalization
    of the learned weights (moving the colour channels independantly in order
    to prevent the network just learning to use chromatic aberation).
    The nine sub-images are then passed passe
    """

    def __init__(self, maxHammingSet, cropSize=225, cellSize=75, tileSize=64,
                 tileLocationRange=10, colourJitter=2):
        """
        cropSize - the size of the square crop used
        cellSize - the dimensions of each subcell of the crop. Dimensions are
        cropSize/3
        tileSize - size of the image cropped from within each cell
        maxHammingSet - 2D array, row is each permutation, column is the elements
        """
        self.cropSize = cropSize
        self.cellSize = cellSize
        self.tileSize = tileSize
        self.colourJitter = colourJitter
        self.tileLocationRange = tileLocationRange
        #  if not maxHammingSet.any():
        #      warnings.warn("Did not pass a set of jigsaw orientations", UserWarning)
        #      temp = list(itertools.permutations(range(9),9))
        #      self.maxHammingSet = np.array(temp[:100], dtype=np.uint8)
        #  else:
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        self.numPermutations = self.maxHammingSet.shape[0]

    def colour_channel_jitter(self, numpy_image):
        """
        Takes in a 3D numpy array and then jitters the colour channels by
        between -2 and 2 pixels (to deal with overfitting to chromatic
        aberations).
        Input - a WxHx3 numpy array
        Output - a (W-4)x(H-4)x3 numpy array (3 colour channels for RGB)
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = numpy_image.shape[0] - self.colourJitter * 2
        y_dim = numpy_image.shape[1] - self.colourJitter * 2
        # Determine the jitters in all directions
        R_xjit = random.randrange(self.colourJitter * 2 + 1)
        R_yjit = random.randrange(self.colourJitter * 2 + 1)
        G_xjit = random.randrange(self.colourJitter * 2 + 1)
        G_yjit = random.randrange(self.colourJitter * 2 + 1)
        B_xjit = random.randrange(self.colourJitter * 2 + 1)
        B_yjit = random.randrange(self.colourJitter * 2 + 1)
        # Seperate the colour channels
        return_array = np.empty((x_dim, y_dim, 3), np.float32)
        for colour_channel in range(3):
            return_array[:,:,colour_channel] = numpy_image[R_xjit:x_dim +
                                            R_xjit, R_yjit:y_dim + R_yjit, colour_channel]
        #  green_channel_array = numpy_image[G_xjit:x_dim +
        #                                    G_xjit, G_yjit:y_dim + G_yjit, 1]
        #  blue_channel_array = numpy_image[B_xjit:x_dim +
        #                                   B_xjit, B_yjit:y_dim + B_yjit, 2]
        # Put the arrays back together and return it
        #  np.stack((red_channel_array, green_channel_array,
        #                   blue_channel_array), axis=-1)
        return return_array

    #  @jit(u1[:](u1[:],u1[:]))
    def create_croppings(self, numpy_array):
        """
        Take in a 3D numpy array and a 4D numpy array containing 9 "jigsaw" puzzles.
        Dimensions of array is 64 (height) x 64 (width) x 3 (colour channels) x 9
        (each cropping)

        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        """
        # Jitter the colour channel
        numpy_array = self.colour_channel_jitter(numpy_array)

        y_dim, x_dim = numpy_array.shape[:2]
        # Have the x & y coordinate of the crop
        crop_x = random.randrange(x_dim - self.cropSize)
        crop_y = random.randrange(y_dim - self.cropSize)
        # Select which image ordering we'll use from the maximum hamming set
        perm_index = random.randrange(self.numPermutations)
        final_crops = np.zeros(
            (self.tileSize, self.tileSize, 3, 9), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                x_start = crop_x + col * self.cellSize + \
                    random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + \
                    random.randrange(self.cellSize - self.tileSize)
                # Put the crop in the list of pieces randomly according to the
                # number picked
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * 3 + col]
                            ] = numpy_array[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
        return final_crops, perm_index



class DataGenerator:
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """

    def __init__(self, maxHammingSet, xDim=64, yDim=64, numChannels=3,
                 numCrops=9, batchSize=32, meanTensor=None, stdTensor=None):
        """
        meanTensor - rank 3 tensor of the mean for each pixel for all colour channels, used to normalize the data
        stdTensor - rank 3 tensor of the std for each pixel for all colour channels, used to normalize the data
        maxHammingSet - a
        """
        self.xDim = xDim
        self.yDim = yDim
        self.numChannels = numChannels
        self.numCrops = numCrops
        self.batchSize = batchSize
        self.meanTensor = meanTensor.astype(np.float32)
        self.stdTensor = stdTensor.astype(np.float32)
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        # Determine how many possible jigsaw puzzle arrangements there are
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        # Use default options for JigsawCreator
        self.jigsawCreator = JigsawCreator(maxHammingSet=maxHammingSet)

    def __data_generation_normalize(self, dataset, batchIndex):
        """
        Internal method used to help generate data, used when
        dataset - an HDF5 dataset (either train or validation)
        """
        # Determine which jigsaw permutation to use
        jigsawPermutationIndex = random.randrange(self.numJigsawTypes)
        x = np.empty((self.batchSize, 256, 256, self.numChannels),
                     dtype=np.float32)
        # creating intermediary numpy array (HDF5 method)
        #  dataset.read_direct(x, source_sel=np.s_[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize, ...])
        x = dataset[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize, ...].astype(np.float32)
        # subtract mean first and divide by std from training set to
        # normalize the image
        x -= self.meanTensor
        x /= self.stdTensor
        # This implementation modifies each image individually
        X = np.empty((self.batchSize, self.xDim, self.yDim,
                      self.numCrops), dtype=np.float32)
        y = np.empty(self.batchSize)
        #  X_i = np.empty((self.xDim, self.yDim, 3, self.numCrops), dtype=np.float32)
        # Python list of 4D numpy tensors for each channel
        X = [np.empty((self.batchSize, self.xDim, self.yDim,
                       self.numChannels), np.float32) for _ in range(self.numCrops)]
        #  pdb.set_trace()
        for image_num in range(self.batchSize):
            # Transform the image into its nine croppings
            single_image, y[image_num] = self.jigsawCreator.create_croppings(
                x[image_num])
            for image_location in range(self.numCrops):
                X[image_location][image_num, :, :, :] = single_image[:, :, :, image_location]
        return X, y

    def sparsify(self, y):
        """
        Returns labels in binary NumPy array
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numJigsawTypes)]
                         for i in range(y.shape[0])])

    #  @threadsafe_generator
    def generate(self, dataset):
        """
        dataset - an HDF5 dataset (either train or validation)
        """
        numBatches = dataset.shape[0] // self.batchSize
        batchIndex = 0
        while True:
             # Load data
            X, y = self.__data_generation_normalize(dataset, batchIndex)
            batchIndex += 1  # Increment the batch index
            if batchIndex == numBatches:
                # so that batchIndex wraps back and loop goes on indefinitely
                batchIndex = 0
            #  pdb.set_trace()
            yield X, self.sparsify(y)
            #  yield X, y
            #  yield X
