
import numpy as np
import cv2
import os
import sys
from descriptor.sift import DoG as dog
import matplotlib.pyplot as plt
#sys.path.append(os.getcwd() + "descriptor/")


def find_maximum_through_scales(pyr,kp,sigma = 1.6 ):
    """
    given a specific keypoints kp (not choosen by the algorithm,but taken from a grid)
    this function aims to find the maximum scale across the octave
    """
    points = [1,2,4,8,16]
    values = []
    for i,p in enumerate(pyr):
        temp_val = np.zeros(5)
        for j in range(5):
            temp_val[j] = p[kp[0]//points[i],kp[1]//points[i],j]
        values.append(temp_val)


    return values


DoG = dog.generate_DoG_pyramid_from_img("../prova1.jpg",4,3,1.6,1.3)






val = find_maximum_through_scales(DoG,(250,250))



val
