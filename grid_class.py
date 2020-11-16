import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import bob
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from scipy.cluster.vq import vq
from descriptor import binary_gabor_features as bgf
from descriptor.co_occurrence_feature import color_hist_vector,Legendre_vector,co_occurrence_vector
from descriptor import co_occurrence_feature as cof
from numpy import asarray
from PIL import Image
import pickle
from grid_classification import utils
from descriptor import binary_gabor_features as bgf
from descriptor.Dense_sift import dog
from descriptor.Dense_sift import generate_descriptor
from descriptor.Dense_sift import find_extrema

from sklearn.metrics import confusion_matrix

"""
RESIZE ALL IMAGE TO 1000*1000
TAKE CENTRAL CROP OF 800*800
TAKE 4 SUB IMAGE OF 400*400--->[100:500,100_500],[100:500,500:900],[500:900,100_500],[500:900,500:900]
FOR EACH OF THIS SUBIMAGES, CALCULATE DENSE SIFT descriptors
K-MEANS CLUSTERING (MAYBE)
64*4 VECTORS
"""














features, label , dic_label= RGB_descriptors(pth)
feat = np.array(features)
label = np.array(label)
#np.save("grid_classification/features/train/ft/dic_label.npy",dic_label)
np.save("grid_classification/features/train/ft/scalesiftdescriptor_200.npy",feat)
np.save("grid_classification/features/train/lab/scalesiftdescriptor_200_lab.npy",label)


test_features, test_label , dic_label_test = RGB_descriptors(test_pth)
test_feat = np.array(test_features)
test_label = np.array(test_label)



#pickle.dump(svc_prec, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_precomputed', 'wb'))


svc_rfb  = OneVsRestClassifier(SVC(kernel = 'rbf',gamma = 'scale'),n_jobs = -1)
svc_rfb = svc_rfb.fit(feat,label)








#qua
svc_pol  = OneVsRestClassifier(SVC(kernel = 'poly',gamma = 'scale'),n_jobs = -1)
svc_pol = svc_pol.fit(feat,label)
pickle.dump(svc_pol, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_pol', 'wb'))

#pickle.dump(svc, open('grid_classification/features/models/color_RGB_200_linear', 'wb'))
svc_sig  = OneVsRestClassifier(SVC(kernel = 'sigmoid',gamma = 'scale'),n_jobs = -1)
svc_sig = svc_sig.fit(feat,label)
pickle.dump(svc_sig, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_sigmoid', 'wb'))




# create confusion matrix





linear_score = svc_rfb.score(test_feat,test_label)
labels = dic_label_test.keys()
pred = svc_rfb.predict(test_feat)
df = utils.evaluated_prediction(pred, test_label, labels)
cm = utils.build_confusion_matrix(df, pred, test_label,labels)
classes = list(dic_label_test.keys())
fig=plt.figure(figsize=(30, 15))
utils.plot_confusion_matrix(cm,classes,"color_RGB_descriptors_rfb_kernel","grid_classification/results/confusion_matrix/",normalize=True,title='Confusion matrix')



models = [svc_prec, svc_lin, svc_rfb, svc_pol,svc_sig]
scalesift_accuracy = []
for i,svc in enumerate(models):
    if(i==0):
        t,_ = test_accuracy(test_features, feat,chisquared_kernel,mean,test_label,svc_prec)
        t = t[0]
        scalesift_accuracy.append(t*100)
    else:
        scalesift_accuracy.append(svc.score(np.array(test_features),np.array(test_label))*100)


scalesift_accuracy
t,_ = test_accuracy(test_features, feat,chisquared_kernel,mean,test_label,svc_prec)

#dense_sift_accuracy
accuracy = np.load("grid_classification/results/color_RGB_accuracy.npy")
accuracy

#dense_sift_accuracy
np.save("grid_classification/results/BGF_color_accuracy",accuracy)






feat = np.load("grid_classification/features/train/ft/color_RGB_200.npy")
label = np.load("grid_classification/features/train/lab/ensesiftdescriptor_200_lab.npy")

test_feat = np.load("grid_classification/features/test/ft/color_RGB_200.npy")
test_label = np.load("grid_classification/features/test/lab/color_RGB_200_lab.npy")"""



feat






from sklearn import metrics
c = metrics.classification_report(test_label, pred,target_names=labels)
