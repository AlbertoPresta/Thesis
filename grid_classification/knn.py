import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from scipy.cluster.vq import vq
from descriptor import binary_gabor_features as bgf

from descriptor import co_occurrence_feature as cof
from numpy import asarray
from PIL import Image
import math
from descriptor.co_occurrence_feature import color_hist_vector,Legendre_vector,co_occurrence_vector
import operator
from descriptor import binary_gabor_features as bgf
from descriptor.Dense_sift import dog
from descriptor.Dense_sift import generate_descriptor
from descriptor.Dense_sift import find_extrema


COORDINATES = [(100,500,100,500),(100,500,500,900),(500,900,100,500),(500,900,500,900)]
CENTERS = [(300,300),(300,700),(700,300),(700,700)]
#CENTERS = [(200,200),(200,400),(400,200),(400,200),(600,600),(600,800),(800,600),(800,800)]

pth = "../final_dataset/train"
test_pth = "../final_dataset/test"
SIZE = 200
lichens = os.listdir(pth)

pth = "../final_dataset/train"
test_pth = "../final_dataset/test"

lichens = os.listdir(pth)






def dense_sift_descriptor(pth):
    sift = cv2.xfeatures2d.SIFT_create()
    lichens = os.listdir(pth)
    features = []
    label = []
    dic_label = {} # salvo corrispondenza classe numero qua
    images = {}
    features_for_lichens = {}
    training_set = {}
    container_features = {}

    l = 0

    for lichen in lichens:
        print("--------- ",lichen," -----------")
        print(l)
        dic_label[lichen] = l
        spec_feat = []
        category = []
        total = []
        lichen_path = os.path.join(pth,lichen)
        lichen_sing_images = os.listdir(lichen_path)
        feature_for_lichens_texture = {}
        for img in lichen_sing_images:
            if(img == ".DS_Store"):
                continue
            img_path = os.path.join(lichen_path,img)
            print(img_path)
            im = Image.open(img_path)
            im = asarray(im)
            im = cv2.resize(im,(1000,1000))

            for ii,crd in enumerate(CENTERS):
                print(crd)
                temporary = []
                crop_img = im[crd[0]-SIZE:crd[0] +SIZE,crd[1] - SIZE:crd[1] +SIZE,:]
                category.append(crop_img)
                kp = cv2.KeyPoint(crd[1],crd[0],SIZE)

                _,feaArrSingle_R = sift.compute(crop_img[:,:,0],[kp])
                _,feaArrSingle_G = sift.compute(crop_img[:,:,1],[kp])
                _,feaArrSingle_B = sift.compute(crop_img[:,:,2],[kp])

                feaArrSingle_R =  feaArrSingle_R.reshape(-1)
                feaArrSingle_G =  feaArrSingle_G.reshape(-1)
                feaArrSingle_B =  feaArrSingle_B.reshape(-1)

                temp = np.concatenate([feaArrSingle_R,feaArrSingle_G,feaArrSingle_B])
                temporary.append(temp)
                features.append(temp)
                spec_feat.append(temp)
                label.append(l)
                total.append([temp,l])
                print("----")
                print(ii)
                print("----")
                feature_for_lichens_texture[img.split(".")[0] + "_crop_ " + str(ii) ] = np.array(temporary)
        features_for_lichens[lichen] = [spec_feat,lichen_sing_images]
        container_features[lichen] = [lichen_sing_images,feature_for_lichens_texture]
        images[lichen] = (category,l)
        l = l + 1

    return features, label , dic_label



def RGB_descriptors(pth):
    lichens = os.listdir(pth)
    features = []
    label = []
    dic_label = {} # salvo corrispondenza classe numero qua

    x_center = np.arange(200,800,400)
    y_center = np.arange(200,800,400)
    l = 0

    for lichen in lichens:
        print("---- ",lichen," ----")
        dic_label[lichen] = l
        lichen_path = os.path.join(pth,lichen)
        lichen_sing_images = os.listdir(lichen_path)
        for img in lichen_sing_images:
            if(img == ".DS_Store"):
                continue
            img_path = os.path.join(lichen_path,img)
            print(img_path)
            im = cv2.imread(img_path)
            #im = asarray(im)
            im = cv2.resize(im,(1000,1000))
            im = im[100:900,100:900,:]
            color = color_hist_vector(im,  bins = 16)
            color = np.asarray(color)
            color = color.reshape(-1)

            for ii,x in enumerate(x_center):
                for jj,y in enumerate(y_center):
                    pt = (x,y)
                    desc = bgf.BGF(im,x,y,SIZE)
                    desc = np.asarray(desc)
                    desc = desc.reshape(-1)

                    temp = np.concatenate([desc,color])

                    features.append(temp)
                    label.append(l)
        l = l + 1
    return features, label, dic_label


def leg_col_co_descriptors(pth):
    lichens = os.listdir(pth)
    features = []
    label = []
    dic_label = {} # salvo corrispondenza classe numero qua

    x_center = np.arange(200,800,400)
    y_center = np.arange(200,800,400)
    l = 0

    for lichen in lichens:
        print("---- ",lichen," ----")
        dic_label[lichen] = l
        lichen_path = os.path.join(pth,lichen)
        lichen_sing_images = os.listdir(lichen_path)
        for img in lichen_sing_images:
            if(img == ".DS_Store"):
                continue
            img_path = os.path.join(lichen_path,img)
            print(img_path)
            im = cv2.imread(img_path)
            #im = asarray(im)
            im = cv2.resize(im,(1000,1000))
            im = im[100:900,100:900,:]

            for ii,x in enumerate(x_center):
                for jj,y in enumerate(y_center):
                    pt = (x,y)
                    print("legendre")
                    legendre = Legendre_vector(im,2)
                    print("histograms")
                    histograms = color_hist_vector(im,  bins = 16)
                    print("co_occurrence")
                    co_occurrence = co_occurrence_vector(im,distance = [1], direction = [0, np.pi/2, 3*np.pi/2])
                    temp = np.concatenate([legendre,histograms,co_occurrence])
                    temp = temp.reshape(-1)

                    features.append(temp)
                    label.append(l)
        l = l + 1
    return features, label, dic_label



def chisquared_distance(hist_1, hist_2):

    k = hist_1.shape[0]

    indexes = np.array([num for num in range(k) if not (hist_1[num] == 0 and hist_2[num] == 0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes])) # the Chi-squared distance
        # plug into the generalized Gaussian kernel
        return 0.5 * D







def scale_dense_sift_descriptors(pth, sigma = 1.6, assumed_blur = 0.5, num_intervals = 3 ,image_border_width = 5, float_tolerance = 1e-7, num_octaves = 3 ):
    lichens = os.listdir(pth)
    features = []
    label = []
    dic_label = {} # salvo corrispondenza classe numero qua

    x_center = np.arange(200,800,400)
    y_center = np.arange(200,800,400)
    l = 0

    for lichen in lichens:
        print("---- ",lichen," ----")
        dic_label[lichen] = l
        lichen_path = os.path.join(pth,lichen)
        lichen_sing_images = os.listdir(lichen_path)
        for img in lichen_sing_images:
            if(img == ".DS_Store"):
                continue
            img_path = os.path.join(lichen_path,img)
            print(img_path)
            im = cv2.imread(img_path,0)
            #im = asarray(im)
            im = cv2.resize(im,(1000,1000))
            im = im[100:900,100:900]

            base_image = dog.generateBaseImage(im, sigma, assumed_blur)
            gaussian_kernels = dog.generateGaussianKernels(sigma, num_intervals)
            gaussian_images = dog.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
            dog_images = dog.generateDoGImages(gaussian_images)

            for ii,x in enumerate(x_center):
                for jj,y in enumerate(y_center):
                    print("---")
                    pt = (x,y)
                    print(pt)
                    kpt, gaussian_index = find_extrema.findscaleExtrema(pt, gaussian_images, dog_images, num_intervals, sigma, image_border_width)
                    octave,layer,scale = find_extrema.unpackOctave(kpt)
                    r = find_extrema.computekeypointwithorientation(kpt,octave,gaussian_images[octave][layer])
                    if(len(r)==0):
                        continue
                    if(len(r)>1):
                        r = r[:1]
                    keypoints = find_extrema.convertKeypointsToInputImageSize(r)
                    desc = generate_descriptor.generateDescriptors(keypoints,gaussian_images)
                    #print(desc.shape)
                    desc = np.array(desc)
                    desc = desc.reshape(-1)
                    features.append(desc)
                    label.append(l)
        l = l + 1
    return features, label, dic_label




def getKNeighbors(trainingSet, testInstance,train_lab, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = np.linalg.norm(trainingSet[x]-testInstance)
        distances.append((trainingSet[x], dist,train_lab[x]))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0],distances[x][2]))
    return neighbors



def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]









def calculate_accuracy_nn(train_data,test_data,train_label,test_label,k = 1):
    acc = []
    for i,tst in enumerate(test_data):

        neigh = getKNeighbors(train_data, tst,train_label, k)
        res = getResponse(neigh)

        if(res==test_label[i]):
            print("ciao")
            acc.append(1)
        else:
            acc.append(0)

    print(len(acc))
    print(np.sum(acc))
    percentuale = np.sum(acc)/len(acc)

    return percentuale*100


features[0][:216].shape

features, label, dic_label =  scale_dense_sift_descriptors(pth)
feat = np.array(features)
test_features, test_label , dic_label_test= scale_dense_sift_descriptors(test_pth)
#features, label, dic_label = calculate_dataset(pth)

len(label)
#test_features, test_label , dic_label_test= calculate_dataset(test_pth)
feat = np.asarray(test_label)
tst = np.asarray(test_features)

acc = calculate_accuracy_nn(feat, tst, label, test_label)

acc

kappas = np.arange(1,201,1)

t = []
for k in kappas:
    print(k)
    acc = calculate_accuracy_nn(feat, tst, label, test_label,k = k)
    t.append(acc)

plt.plot(t)
np.max(t)

"""rows = ['BGF','COL_BGF','DENSE_SIFT','SCALE_SIFT']
columns = ['svm-prec','svm-lin','svm-rfb','svm-pol','svm-sig','nearest-neigh','knn']
cell_text =[["22","20.8","21.3","6.4","12.9","24.53","25.46-4"],
            ["30.01","20.3","20.37","12.5","20.37","23.14","25.4-4"],
            ["14.35","11.57","11.57","12.03","3.7","12.5","12.5-1"],
            ["7.97","9,81","7.97","7.98","6.1","6.7","6.7-1"]]
# Add a table at the bottom of the axes
%matplotlib inline
fig, ax = plt.subplots()
fig=plt.figure(figsize=(8, 8))


plt.table(cellText=cell_text,rowLabels=rows,colLabels=columns,loc='center')
fig.tight_layout()
plt.savefig("prova")
plt.show()"""
