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
import os


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



    def morphological_cv(self, size_checkerboard = 6 , number_of_iterations = 10):
        # Initial level set
        init_ls = checkerboard_level_set(self.gray_image.shape, size_checkerboard)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(self.gray_image, number_of_iterations, init_level_set=init_ls, smoothing=3,iter_callback=callback)
        return ls, evolution



    def morphological_gac(self):
        gimage = inverse_gaussian_gradient(self.gray_image)
        # Initial level set
        init_ls = np.zeros(self.gray_image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_geodesic_active_contour(gimage, 230, init_ls, smoothing=1, balloon=-1, threshold=0.69, iter_callback=callback)
        return ls, evolution
