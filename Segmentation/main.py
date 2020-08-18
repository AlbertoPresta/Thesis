import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append(os.getcwd() + "Segmentation/")
import Segmentation.Segmenter as segmenter
import numpy as np
import matplotlib.pyplot as plt
import time as time
from skimage.segmentation import mark_boundaries





s = segmenter.Segmenter("Segmentation/images/vip6.jpg")
c,d,e = s.Otzu_thresholding()
ls, evolution = s.morphological_cv(number_of_iterations = 35)  # Active Contours Without Edges
# print result of morhological_cv
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
ax = axes.flatten()
ax[0].imshow(s.image, cmap="gray")
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
plt.savefig("Segmentation/results/_" + s.image_name + "_morhological_cv_segmentation.jpg")
