import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from scipy.io import savemat,loadmat
import pickle
from scipy.io import loadmat
import numpy as np
os.listdir('grid_classification/')

def chisquared_kernel(hist_1,hist_2,a):

    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num]==0 and hist_2[num]==0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes]))
        return np.exp(- (D * 0.5) / a)


descriptor_path = 'grid_classification/im_test/descr/bag_400_8/text_image_N4_desc'
label_path = 'grid_classification/im_test/labels'
img_path = 'grid_classification/im_test/images'

loaded_model = pickle.load(open('grid_classification/model/finalized_model.sav', 'rb'))
mean = pickle.load(open('grid_classification/model/mean.sav', 'rb'))

training = pickle.load(open('grid_classification/model/training.sav', 'rb'))
num_objects = training.shape[0]
def prova(desc_path,model):
    dsc = os.path.join(desc_path, 'descriptors.mat')
    dsc = loadmat(dsc)
    dsc = dsc['descriptors']
    res = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            temp_dsc = dsc[i,j,:]
            c = model.predict(np.array([chisquared_kernel(temp_dsc, training[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
            res[i,j] = c[0]
    return res



c = prova(descriptor_path, loaded_model)

c
f = np.load('grid_classification/im_test/labels/text_image_N4.npy')
f
