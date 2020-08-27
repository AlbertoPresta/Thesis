import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import filters









image = cv2.imread('Segmentation/images/vip3.jpg').astype(np.float32) / 255
c = filters.filter_with_laplacian_of_gaussian(image, 0, 'reflect' , 0 )

plt.imshow(c)

one_hot_pix_b.shape

one_hot_pix_b = np.zeros((image.shape[0]*image.shape[1], 11), dtype=np.int32)
one_hot_pix_b[np.arange(image.shape[0]*image.shape[1]), np.floor(image).flatten())] = 1



plt.imshow(image)

list_of_kernels = [('log',.1,'reflect',0),('log',.2,'reflect',0),('gabor',(10, 10), 5, 45, 10, 1, 0, cv2.CV_32F),('gabor',(5, 5), 5, 30, 10, 1, 0, cv2.CV_32F)]
images = filters.image_filtering(image, list_of_kernels)
g = filters.gabor_kernel((10, 10), 5, 45, 10, 1, 0, cv2.CV_32F)
dst = filters.filter_with_gabor_kernel(image,g)


plt.imshow(images[0])

plt.figure(figsize = (8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap = 'gray')
plt.subplot(132)
plt.axis('off')
plt.title('kernel')
plt.imshow(kernel, cmap = 'gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap = 'gray')
