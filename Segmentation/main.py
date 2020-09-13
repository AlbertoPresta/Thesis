import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append(os.getcwd() + "Segmentation/")
import Segmentation.Segmenter as segmenter
import numpy as np
import matplotlib.pyplot as plt
import time as time
from skimage.segmentation import mark_boundaries


img = os.listdir("super/")
img = ["super/" + i for i in img]

img

# normal chan vese algorithm
for l in img:
    for i in os.listdir(l):
        print(l + "/" + i)
        s = segmenter.Segmenter(l + "/" + i)
        cv = s.morphological_cv(number_of_iterations = 20)





# normal chan vese algorithm
for l in img:
    for i in os.listdir(l):
        print(l + "/" + i)
        s = segmenter.Segmenter(l + "/" + i)
        cv = s.Otzu_thresholding()



for im in img:
    print(im)
    s = segmenter.Segmenter(im)
    cv = s.chan_vese()




for im in img:
    print(im)
    s = segmenter.Segmenter(im)
    cv = s.kmeans(2)



for im in img:
    print(im)
    s = segmenter.Segmenter(im)
    s.mean_shift()
