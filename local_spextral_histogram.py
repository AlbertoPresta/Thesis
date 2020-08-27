import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg as LAsci
import math
from skimage import io, color



def local_spectral_histogram(Igs, ws, Bins ):
    """
    functionwhich computes local spectral hitogram of a figure.
    IGS: a n-band images, the first one is the original image, the rest are filtered version of an image
    ws: window size
    Bins: number of bins of the histogram
    """
    h, w, bn = Ig.shape #h = height of images, w = width of images, bn = number of images

    #quantize values at each pixel into bin ID
    for i in range(bn):
        b_max = np.max(Ig[:, :, i]) # maximum value of pixel
        b_min = np.min(Ig[:, :, i]) # minimum value of pixel
        assert b_max != b_min, "Band %d has only one value!" % i
        b_interval = (b_max - b_min) * 1. / Bins  # interval of bins 
        Igs[:, :, i] = np.floor((Ig[:, :, i] - b_min) / b_interval)

    Ig[Ig >= BinN] = BinN-1
    Ig = np.int32(Ig)

    # convert to one hot encoding
    one_hot_pix = []

    
