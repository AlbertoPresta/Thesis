import cv2
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from descriptor import binary_gabor_features as binary_gabor_features
from descriptor import distance
from descriptor import co_occurrence_feature as co_occurrence_feature
from descriptor import surf
import operator
from scipy.stats import wasserstein_distance
from descriptor.sift import DoG as dog
%matplotlib inline




refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    image_pt = param[0]
    img = cv2.imread(image_pt)
    size = param[1]
    hlf = param[2]
    feature = param[3]
    dg = param[4]
    sig = param[5]
    gaussian_images = param[6]
    global refPt,cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        print(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        print(x, y)
        xa,ya = refPt[0]
        xb,yb = refPt[1]
        img_color = cv2.imread(image_pt)
        imga = img_color[ya - size:ya + size,xa-size:xa + size,:]
        imgb = img_color[yb - size:yb + size,xb-size:xb + size,:]


        if(feature == "LBP"):
            feata = LBP.LBP_features(image_pt,True,sub = True, coord = [xa-25, xa + 25, ya -25,ya + 25])[0]
            binsa = np.arange(0,243)
            weightsa = np.ones(len(feata)) / len(feata)
            p_a,_ = np.histogram(feata,binsa,weights= weightsa)
            featb = LBP.LBP_features(image_pt,True,sub = True, coord = [xb-25, xb + 25, yb -25,yb + 25])[0]
            binsb = np.arange(0,243)
            weightsb = np.ones(len(featb)) / len(featb)
            p_b,_ = np.histogram(featb,binsb,weights= weightsb)
            distanza = wasserstein_distance(p_a,p_b)
            print("wasserstein_distance ",refPt[0]," and ",refPt[1]," is: ",distanza)
            refPt = []
        elif(feature == "legendre-color-co_occurrence"):
            feat_a = co_occurrence_feature.total_feature(imga,50)
            feat_b = co_occurrence_feature.total_feature(imgb,50)

            distanza = distance.Eucledian_distance(feat_a,feat_b)
            print("Eucledian_distance ",refPt[0]," and ",refPt[1]," is: ",distanza)
            refPt = []
        elif(feature == "gabor"):
            gabor_a = binary_gabor_features.BGF(img, xa, ya, size, halflngt = hlf)
            gabor_b = binary_gabor_features.BGF(img, xb, yb, size, halflngt = hlf)
            distanza = distance.Eucledian_distance(gabor_a,gabor_b)
            print("Eucledian_distance ",refPt[0]," and ",refPt[1]," is: ",distanza)
            refPt = []
        elif(feature == "surf"):



            print("sono qua ")
            surf_a = surf.calculate_surf_in_a_specific_point(image_pt,(xa,ya),dg,sig,gaussian_images)
            print("fatto")
            surf_b = surf.calculate_surf_in_a_specific_point(image_pt,(xb,yb),dg,sig,gaussian_images)
            distanza = distance.Eucledian_distance(surf_a,surf_b)
            print("Eucledian_distance ",refPt[0]," and ",refPt[1]," is: ",distanza)
            refPt = []





# construct the argument parser and parse the arguments

# load the image, clone it, and setup the mouse callback function
image = cv2.imread("prova3.jpg")
image = cv2.resize(image, (512,512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sig = dog.generate_sigmas()
dg, gaussian_images = dog.dogs(gray,sig)
clone = image.copy()
cv2.namedWindow("image")
param = ["prova3.jpg",25,7,"surf",dg,sig,gaussian_images]
cv2.setMouseCallback("image", click_and_crop,param)
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
