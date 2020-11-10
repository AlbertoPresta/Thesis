import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import cv2
import itertools



def read_and_process_images(list_of_images,labels,dimension=64):
    """
    Functions which creates an array containing a list og 64x64 images
    Input List_of_images : array of strings with patterns of all images (obtained with list_of_path function)
    Input Dimension(64): integer which represents dimension of the image

    Output x = array of cv2  normalized images
    Output y = array of labels related to images
    """
    x = [] #array of images
    y = [] #array of labels
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image,cv2.IMREAD_GRAYSCALE),(dimension,dimension),interpolation=cv2.INTER_CUBIC))
        y.append(labels.index(image.split("/")[3]))

    x = np.asarray(x)
    y = np.asarray(y)
    x = x/255
    return x,y



def list_of_path(lab,path,shuf = True):
    """
    Function which creates a list of strings, each of them represents the pattern of a specific
    image contained in path
    Input lab = array of strings contatining all the labels of the classficiation problem
    Input path = string represented location of the images
    Input Shuf(True) = boolean which allows to mix results

    Output x = array with all the patterns of images
    """
    x = []
    for i in lab:
        s = path+i+'/{}'
        temp = [s.format(i) for i in os.listdir(path+i+'/')]
        x = x + temp
    if(shuf==True):
        x = np.random.permutation(x)
    return x
