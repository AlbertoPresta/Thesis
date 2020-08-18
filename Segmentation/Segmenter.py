import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time as time
import math
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import data, img_as_float
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
import os
from skimage import data
from skimage import filters
from skimage import exposure


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


class Segmenter:

    def __init__(self, image_pt):
        self.image_name = os.path.basename(image_pt).split(".")[0]
        self.image_pt = image_pt
        self.image = img_as_float(Image.open(image_pt))
        self.gray_image = rgb2gray(self.image)



    def Otzu_thresholding(self ):
        #this function also print and save results
        val = filters.threshold_otsu(self.gray_image)
        hist, bins_center = exposure.histogram(self.gray_image)
        plt.figure(figsize=(9, 4))
        plt.subplot(131)
        plt.axis('off')
        plt.imshow(self.image, cmap='gray', interpolation='nearest')
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(self.gray_image, cmap='gray', interpolation='nearest')
        plt.subplot(133)
        plt.imshow(self.gray_image < val, cmap='gray', interpolation='nearest')
        title = "Otzu Thresholding"
        plt.savefig("Segmentation/results/_" + self.image_name + "_otzu_thresholding.jpg")
        return val, hist, bins_center


    def morphological_cv(self, size_checkerboard = 6 , number_of_iterations = 10):
        # Initial level set
        init_ls = checkerboard_level_set(self.gray_image.shape, size_checkerboard)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(self.gray_image, number_of_iterations, init_level_set=init_ls, smoothing=3,iter_callback=callback)
        return ls, evolution


    # DOES NOT WORK GOOD!
    def morphological_gac(self):
        gimage = inverse_gaussian_gradient(self.gray_image)
        # Initial level set
        init_ls = np.zeros(self.gray_image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_geodesic_active_contour(gimage, 230, init_ls, smoothing=1, balloon=-1, threshold=0.69, iter_callback=callback)
        return ls, evolution



    def quickshift(self, kernel_size=3, max_dist=60000, ratio=0.5):
        res = quickshift(self.image, kernel_size = kernel_size, max_dist=max_dist, ratio=ratio)
        return res
