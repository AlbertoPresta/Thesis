
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
import cv2
import math



def filter_with_gabor_kernel(image, gabor):
    filtered = cv2.filter2D(image, -1, gabor)
    return filtered


def gabor_kernel(ksize, sigma, theta, lam, gamma, psi, ktype = cv2.CV_32F):
    kernel = cv2.getGaborKernel(ksize, sigma, theta, lam, gamma, psi, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    return kernel



def filter_with_laplacian_of_gaussian(image, sig, md , val ):
    res = ndi.gaussian_laplace(image, sigma = sig, mode = md, cval = val)
    return res



def image_filtering(image, filter_list):
    sub_img = []
    for filter in filter_list:
        print(filter)
        assert (filter[0] == 'log') | (filter[0] == 'gabor'), 'Undefined filter name. '
        if filter[0] == 'log':
            r = filter_with_laplacian_of_gaussian(image,filter[1],filter[2],filter[3])
            sub_img.append(r)
        elif filter[0] == 'gabor':
            f = gabor_kernel(filter[1], filter[2], filter[3], filter[4], filter[5],filter[6])
            r = filter_with_gabor_kernel(image,f)
            sub_img.append(r)
    return sub_img


from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
img = io.imread('Segmentation/images/vip.jpg')

c = filter_with_laplacian_of_gaussian(img, .5, "reflect" , 0 )

plt.imshow(c)
