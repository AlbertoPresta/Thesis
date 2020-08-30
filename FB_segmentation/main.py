import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import math
sys.path.append(os.getcwd() + "FB_segmentation/")
import FB_segmentation.filters as filters
import matplotlib.pyplot as plt
import FB_segmentation.Factor_based_segmentation as fbseg







filter_list = [('log', .5,[3,3]), ('log', 1,[5,5]),
                   ('gabor', 1.5, 0), ('gabor', 1.5, math.pi/2), ('gabor', 1.5, math.pi/4), ('gabor', 1.5, -math.pi/4),
                   ('gabor', 2.5, 0), ('gabor', 2.5, math.pi/2), ('gabor', 2.5, math.pi/4), ('gabor', 2.5, -math.pi/4)
                   ]



p = filters.filters("Segmentation/images/vip5.jpg",filter_list)

Ig = p.image_filtering()

plt.imshow(Ig[:,:,0])

a = fbseg.FB_seg(Ig,"Segmentation/images/vip5.jpg")

seg_out = a.Fseg()

a.plot_and_save_results(seg_out)
