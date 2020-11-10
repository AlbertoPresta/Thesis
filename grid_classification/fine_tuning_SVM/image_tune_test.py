import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras import models
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

model = load_model("grid_classification/fine_tuning_SVM/my_model")



def divide_image_in_patches(img_pth,model):
    img = Image.open(img_pth)
    img = np.asarray(img)
    img = cv2.resize(img,(1000,1000))
    griglia_feat = []
    x_center = np.arange(50,1050,100)
    y_center = np.arange(50,1050,100)
    SIZE = 50
    for i,x in enumerate(x_center):
        print(i)
        ft_temp = []
        for j,y in enumerate(y_center):
            im = img[x-SIZE:x+SIZE,y-SIZE:y+SIZE,:]
            cv2.imwrite("temp.jpg",im)
            ft_temp.append(classify_cell_with_model("temp.jpg",model))
        griglia_feat.append(ft_temp)
    return griglia_feat



def classify_cell_with_model(pth,model):
    temp_img=image.load_img(pth,target_size=(100,100))
    temp_img=image.img_to_array(temp_img)
    temp_img = preprocess_input(temp_img)
    res = []
    res.append(temp_img)
    res = np.array(res)
    pred = model.predict_classes(res)
    return pred[0]

import numpy as np
griglia = divide_image_in_patches("immagini_test/prova3.jpg",model)



def create_masks(griglia, im,name,direc = "grid_classification/results/test_masks/image_" ):
    uniqueValues, occurCount = np.unique(griglia, return_counts=True)
    for val in uniqueValues:
        mask = copy.deepcopy(im)
        for i in range(griglia.shape[0]):
            for j in range(griglia.shape[1]):
                if(griglia[i,j]==val):
                    for ii in range(i*100,(i+1)*100):
                        for jj in range(j*100,(j+1)*100):
                            mask[ii,jj,:] = [0,0,0]
        plt.imshow(mask)
        plt.title("image prova3 class " + str(val))
        plt.savefig(direc + name + "_class" + str(val) + ".jpg")


import copy
im = Image.open("immagini_test/prova3.jpg")
im = np.asarray(im)
im = cv2.resize(im,(1000,1000))
griglia = np.array(griglia)
create_masks(griglia, im, "image1")
