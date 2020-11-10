import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import  Image
import cv2
from descriptor.Dense_sift import dog
from descriptor.Dense_sift import generate_descriptor
from descriptor.Dense_sift import find_extrema
import numpy as np


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from scipy.cluster.vq import vq
from numpy import asarray
from PIL import Image


pth = "../final_dataset/train"
test_pth = "../final_dataset/test"



# calculate precision/recall
