import os
import matplotlib.pyplot as plt
import PIL
from PIL import  Image
import cv2
from descriptor.Dense_sift import dog
from descriptor.Dense_sift import generate_descriptor
from descriptor.Dense_sift import find_extrema
import numpy as np









image = cv2.imread("prova.jpg",0)
image = cv2.resize(image,(1000,1000))
sigma = 1.6
assumed_blur = 0.5
num_intervals = 3
image_border_width=5
float_tolerance = 1e-7


base_image = dog.generateBaseImage(image, sigma, assumed_blur)
num_octaves = 3
gaussian_kernels = dog.generateGaussianKernels(sigma, num_intervals)
gaussian_images = dog.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
dog_images = dog.generateDoGImages(gaussian_images)

x_center = np.arange(16,800,1


image = image.astype('float32')


%%time
for ii,x in enumerate(x_center):
    for jj,y in enumerate(y_center):
        print("-----")
        print(x,":",y)
        kpt, gaussian_index = find_extrema.findscaleExtrema(pt, gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        octave,layer,scale = find_extrema.unpackOctave(kpt)
        r = find_extrema.computekeypointwithorientation(kpt,octave,gaussian_images[octave][layer])
        if len(r)>1:
            print(x,",",y)
        keypoints = find_extrema.convertKeypointsToInputImageSize(r)
        desc = generate_descriptor.generateDescriptors(keypoints,gaussian_images)


desc.shape
