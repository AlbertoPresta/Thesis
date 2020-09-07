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




images = os.listdir("Segmentation/images/")
images = ["Segmentation/images/" + im for im in images]



for im in images:
    print(im)
    p = filters.filters(im,filter_list)
    Ig = p.image_filtering()
    print("image ",im.split("/")[-1], " filtered")
    a = fbseg.FB_seg(Ig,im)
    seg_out = a.Fseg()
    print("image ",im.split("/")[-1]," segmented")
    a.plot_and_save_results(seg_out)






a = fbseg.FB_seg(Ig,"Segmentation/images/vip.jpg")



seg_out = a.Fseg()

a.plot_and_save_results(seg_out)
