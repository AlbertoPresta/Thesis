import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy


c = np.zeros((4,4,3))

plt.imshow()


def test_accuracy(test_features,features,kernel,mean, test_label,svc):
    num_objects = features.shape[0]
    res = []
    prediction = []
    for i,ft in enumerate(test_features):
        pred = svc.predict(np.array([kernel(ft, features[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
        prediction.append(pred[0])
        #print(i,": ",pred," : ",test_label[i])
        if(pred==test_label[i]):
            res.append(1)
        else:
            res.append(0)

    res = np.array(res).reshape(-1)
    return np.sum(res)/res.shape, np.array(prediction)



def test_accuracy(svc, test_features, test_label):
    return svc.score(np.array(test_features),np.array(test_label))




def divide_image_in_crops(I):

    # place where to save crops of the image
    res = {}

    centers = np.arange(50,1050,100)

    r = 50

    for i,x in enumerate(centers):
        for j,y in enumerate(centers):
            res[(i,j)] = I[x - r :x + r ,y - r : y + r,:]
    return res







def extract_accuracy_on_test_image(I,descriptor_pth, classifier, true_classification):

    TP = {}
    TN = {}
    FP = {}
    FN = {}
    lich_in_image = np.unique(true_classification)
    descriptors = loadmat(descriptor_pth)['descriptors']
    (m,n,visual_words) = descriptors.shape
    pred = svc_rfb.predict(test_feat)
    cl = np.zeros(m,n)
    # applichiamo il classficatore svm suicrops
    for i in range(m):
        for j in range(n):
            dsc = descriptors[i,j,:]
            cl[i,j] = classifier.predict(dsc)


    # create mask
    mask = np.zeros((1000,1000,3)).astype(np.uint8)
    mask[0:500,0:500,:] = [255,0,0]
    mask[0:500,500:1000,:] = [0,255,0]
    mask[500:1000,0:500,:] = [0,0,255]
    mask[500:1000,500:1000,:] = [255,255,0]

    # creo dictionary con colory importanti

    l1 = true_classification[0,0]
    l2 = true_classification[0,5]
    l3 = true_classification[5,0]
    l4 = true_classification[9,9]

    d = {}
    d[l1] = [255,0,0]
    d[l2] = [0,255,0]
    d[l3] = [0,0,255]
    d[l4] = [255,255,0]



    mask_I = I.copy()

    for i in range(m):
        for j in range(n):
            if cl[i,j] == l1:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100),((j+1)*100),d[l1],-1)
            elif cl[i,j] == l2:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100),((j+1)*100),d[l2],-1)
            elif  cl[i,j] == l3:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100),((j+1)*100),d[l3],-1)
            elif cl[i,j] == l4:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100),((j+1)*100),d[l4],-1)
            else:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100),((j+1)*100),(125,125,125),-1)


    fig=plt.figure(figsize=(30, 15))
    f, axarr = plt.subplots(1,3)
    axarr[0,0] = plt.imshow(I)
    axarr[0,1] =plt.imshow(mask)
    axarr[0,2] = plt.imshow(mask_I)
