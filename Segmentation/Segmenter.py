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
from skimage.segmentation import chan_vese


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


    def plot_image(self):
        plt.imshow(self.image)

    def plot_gray_image(self):
        plt.imshow(self.gray_image)

    def Otzu_thresholding(self, save = True ):
        #this function also print and save results
        val = filters.threshold_otsu(self.gray_image)
        hist, bins_center = exposure.histogram(self.gray_image)
        if(save):
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
            plt.savefig("Segmentation/results/otzu/_" + self.image_name + "_otzu_thresholding.jpg")
        return val, hist, bins_center



    def chan_vese(self,mu = 0.25, lam1 = 1, lam2 = 1, tol = 1e-3, max_iter = 200, dt = 0.5, initial = "checkerboard", ext_out = True, save = True ):
        cv = chan_vese(self.gray_image, mu=mu, lambda1=lam1, lambda2=lam2, tol = tol, max_iter=max_iter, dt= dt, init_level_set= initial, extended_output = ext_out)

        # plot and save results:
        if(save):
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            ax = axes.flatten()

            ax[0].imshow(self.image)
            ax[0].set_axis_off()
            ax[0].set_title("Original Image", fontsize=12)

            ax[1].imshow(cv[0], cmap="gray")
            ax[1].set_axis_off()
            title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
            ax[1].set_title(title, fontsize=12)

            ax[2].imshow(cv[1], cmap="gray")
            ax[2].set_axis_off()
            ax[2].set_title("Final Level Set", fontsize=12)

            ax[3].plot(cv[2])
            ax[3].set_title("Evolution of energy over iterations", fontsize=12)

            fig.tight_layout()
            plt.savefig("Segmentation/results/chan_vese/_" + self.image_name + "_.jpg")
            plt.show()

            #plt.savefig("Segmentation/results/chan_vese/_" + self.image_name + "_.jpg")
        return cv



    def morphological_cv(self, size_checkerboard = 6 , number_of_iterations = 10, save = True):
        # Initial level set
        init_ls = checkerboard_level_set(self.gray_image.shape, size_checkerboard)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(self.gray_image, number_of_iterations, init_level_set=init_ls, smoothing=3,iter_callback=callback)
        if (save == True):
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            ax = axes.flatten()
            ax[0].imshow(self.image, cmap="gray")
            ax[0].set_axis_off()
            ax[0].contour(ls, [0.5], colors='r')
            ax[0].set_title("Morphological chan-vese segmentation", fontsize=12)
            ax[1].imshow(ls, cmap="gray")
            ax[1].set_axis_off()
            contour = ax[1].contour(evolution[1], [0.5], colors='g')
            contour.collections[0].set_label("Iteration 2")
            contour = ax[1].contour(evolution[-1], [0.5], colors='r')
            contour.collections[0].set_label("Last iteration")
            ax[1].legend(loc="upper right")
            title = "Morphological chan-vese evolution"
            ax[1].set_title(title, fontsize=12)
            plt.savefig("Segmentation/results/morphological_chan_vese/_" + self.image_name + "_.jpg")
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



    def quickshift(self, kernel_size=3, max_dist=60, ratio=0.5):
        res = quickshift(self.image, kernel_size = kernel_size, max_dist=max_dist, ratio=ratio)
        return res
