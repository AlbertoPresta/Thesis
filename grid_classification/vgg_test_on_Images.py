import pandas as pd
import matplotlib.pyplot as plt;
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical #to create dummy variable
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.svm import SVC
import warnings
import operator
import cv2
import copy
import PIL
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel,chi2_kernel
from PIL import Image
warnings.filterwarnings('ignore')

model = VGG16(weights='imagenet', include_top=False)

def feature_extraction(dataset,pre_model):
    """
    Function which extract feature of a dataset fro the last convolutional layer of the pre_model

    Input dataset= path of the train images
    Input pre_model = pre-trained model for extracting feature

    Output  images = images of the dataset
    Output feature = feature with shape (7,7,512)
    Output feature_flatted = flattened features
    """
    res = []
    cont = 0
    #viaggio nel dataset
    for path in dataset:
        cont = cont+1
        im = load_img(path, target_size=(224,224))
        if(cont==1):
            print(im.size)
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        im = imagenet_utils.preprocess_input(im)
        res.append(im)
    images = np.vstack(res)
    features = pre_model.predict(images, batch_size=64)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return  features_flatten

im = Image.open("immagini_test/prova1.jpg")
im = np.asarray(im)
im = cv2.resize(im,(1000,1000))
plt.imshow(im)

def divide_image_in_patches(img_pth,model):
    img = cv2.imread(img_pth)
    img = cv2.resize(img,(1000,1000))

    griglia_feat = []

    x_center = np.arange(50,1050,100)
    y_center = np.arange(50,1050,100)
    SIZE = 50
    for i,x in enumerate(x_center):
        ft_temp = []
        for j,y in enumerate(y_center):
            im = img[x-SIZE:x+SIZE,y-SIZE:y+SIZE,:]
            cv2.imwrite("temp.jpg",im)
            ft_temp.append(feature_extraction(["temp.jpg"],model).reshape(-1))
        griglia_feat.append(ft_temp)
    return griglia_feat

%%time
vectors = divide_image_in_patches("immagini_test/prova1.jpg",model)







def classify_each_cell(vectors,svm = svm):
    griglia = np.zeros((10,10))

    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
            v = vectors[i,j]
            x = svm.predict([v])
            griglia[i,j] = x[0]
    return griglia







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




%%time
im = Image.open("immagini_test/prova3.jpg")
im = np.asarray(im)
im = cv2.resize(im,(1000,1000))
vectors = divide_image_in_patches("immagini_test/prova3.jpg",model)
vectors = np.array(vectors)
griglia = classify_each_cell(vectors)
create_masks(griglia, im, "image1")
