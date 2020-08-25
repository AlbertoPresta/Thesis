import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel




def gabor_filter(freq,theta, sig_x,sig_y):
     kernel = np.real(gabor_kernel(freq, theta=theta,sigma_x=sig_x, sigma_y=sig_y))
     return kernel



def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    filtered_images = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        filtered_images.append(filtered)
    return feats, filtered_images
