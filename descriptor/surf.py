import cv2
from descriptor.sift import DoG as dog
from scipy.ndimage import gaussian_filter
from descriptor.sift.sift_prova import assign_orientation
import matplotlib.pyplot as plt
import os
import numpy as np



def crop_image(img_path,new_width = 400, new_height = 400):
    img = cv2.imread(img_path)
    print("original_shape: ",img.shape)
    img = cv2.resize(img,(1000,1000))
    #plt.imshow(img)
    #print(img.shape)
    height, width,_ = img.shape
    height = height//2
    width = width//2

    left = (width - new_width)
    top = (height - new_height)
    right = (width + new_width)
    bottom = (height + new_height)
    print(left,right,top,bottom)

    img = img[top:bottom,left:right,:]
    return img



def mean_vector(surf_feat):
    mean = []
    for i in range(surf_feat.shape[2]):
        tmp_mean = []
        for j in range(surf_feat.shape[0]):
            tmp_mean.append(surf_feat[j][0][i])
        mean.append(np.mean(tmp_mean))
    return np.array(mean)


def create_surf_descriptors(img,step_size = 20, thresh = 400):
    #img = crop_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(thresh)
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(10, gray.shape[0], step_size)
                  for x in range(10, gray.shape[1], step_size)]

    # generate dogs to calculate best scale (question about this)
    sig = dog.generate_sigmas()
    dg, gaussian_images = dog.dogs(gray,sig)

    surfs = []
    for k in kp:
        x = int(k.pt[1])
        y = int(k.pt[0])
        sigma, index = dog.maximum(dg,sig,(x,y))
        _,c = surf.compute(gaussian_images[index],[k])
        surfs.append(c)
    surfs = np.array(surfs)
    mean = mean_vector(surfs)
    return surfs, mean





def calculate_surf_in_a_specific_point(img_path,kp,dg,sig,gaussian_images,thresh = 400):

    cx = kp[0]
    cy = kp[1]
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(thresh)
    k = cv2.KeyPoint(cx, cy,20)
    # generate dogs to calculate best scale (question about this)
    sigma, index = dog.maximum(dg,sig,(cx,cy))
    # generate dogs to calculate best scale (question about this)
    _,c = surf.compute(gaussian_images[index],[k])
    return c










#image = cv2.imread("../prova2.jpg",0)
#surf_feat = create_surf_descriptors(image)
#surf_mean = mean_vector(surf_feat)
