import cv2
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
from grid_classification.grid_class import compute_gram_matrix
from descriptor import binary_gabor_features as bgf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from descriptor.co_occurrence_feature import color_hist_vector,Legendre_vector,co_occurrence_vector
import copy
from keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False)


feat = np.load("grid_classification/features/train/ft/color_RGB_200.npy")
label = np.load("grid_classification/features/train/lab/color_RGB_200_lab.npy")

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
    return images, features, features_flatten

#test_feat = np.load("grid_classification/features/test/ft/color_RGB_200.npy")
#test_label = np.load("grid_classification/features/test/lab/color_RGB_200_lab.npy")

def compute_gram_matrix(data,kernel):
    samples,_ = np.shape(data)
    matrix = np.zeros((samples,samples))
    for r in range(samples):
        print(r)
        for c in range(r,samples):
            tmp = kernel(data[r],data[c])
            matrix[r,c] = tmp
            matrix[c,r] = tmp
    return matrix


def chisquared_kernel(hist_1,hist_2,a):

    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num]==0 and hist_2[num]==0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes]))
        return np.exp(- (D * 0.5) / a)






def chisquared_distance(hist_1, hist_2):

    k = hist_1.shape[0]

    indexes = np.array([num for num in range(k) if not (hist_1[num] == 0 and hist_2[num] == 0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes])) # the Chi-squared distance
        # plug into the generalized Gaussian kernel
        return 0.5 * D
# calcolo il modello di svm


%%time
#feat = np.array(features)
gram= compute_gram_matrix(feat,chisquared_distance)
mean = np.mean(gram[np.triu_indices(np.shape(feat)[0])])
gram = np.exp(-(gram/mean)) # generalized Gaussian kernel
svc_prec  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
svc_prec = svc_prec.fit(gram,np.array(label))



image_path = "immagini_test/"
img_pth = os.path.join(image_path,"prova3.jpg")

img = Image.open(img_pth)
img = img.resize((1000,1000))
img = np.asarray(img)
x_center = np.arange(50,1050,100)
x_center

def create_griglia(svc, feat, image,kernel):
    SIZE = 50
    griglia = np.zeros((10,10))

    x_center = np.arange(50,1050,100)
    y_center = np.arange(50,1050,100)
    num_objects = feat.shape[0]

    for i,x in enumerate(x_center):
        for j,y in enumerate(y_center):
            #creo descr ittore
            print("-----")
            print(i,j)
            imm = image[x-SIZE:x+SIZE,y-SIZE:y+SIZE,:]
            desc = bgf.BGF(imm,x,y,SIZE,all = True)
            color = color_hist_vector(imm,  bins = 16)
            color = np.array(color)
            color = color.reshape(-1)
            desc = np.array(desc)
            desc = desc.reshape(-1)
            temp = np.concatenate([desc,color])
            pred = svc.predict(np.array([kernel(temp, feat[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
            print("end prediction")
            griglia[i,j] = pred[0]

    # refine griglia
    print("refine griglia")
    for i in range(10):
        for j in range(10):
            if(i==0 and j==0 and griglia[i+1,j]==griglia[i,j+1]):
                griglia[i,j] = griglia[i+1,j]
            elif(i==0 and j==9 and griglia[i+1,j]==griglia[i-1,i-1] == griglia[i,j-1] and griglia[i,j]!= griglia[i,j-1]):
                griglia[i,j] = griglia[i-1,j-1]
            elif(i==0 and 0<j<9 and griglia[i,j-1]==griglia[i+1,j]==griglia[i,j+1]):
                griglia[i,j] = griglia[i,j-1]
            elif(j==0 and 0<i<9 and griglia[i-1,j]==griglia[i,j+1]==griglia[i+1,j]):
                griglia[i,j] = griglia[i-1,j]
            elif(j==9 and 0<i<9 and griglia[i-1,j]==griglia[i,j-1]==griglia[i+1,j]):
                griglia[i,j] = griglia[i-1,j]
            elif(0<i<9 and 0<j<9 and griglia[i-1,j]==griglia[i,j-1]==griglia[i+1,j]==griglia[i,j+1]):
                griglia[i,j] = griglia[i-1,j]
            elif(i==9 and 0<j<9 and griglia[i,j-1]==griglia[i-1,j]==griglia[i,j+1]):
                griglia[i,j] = griglia[i,j-1]




    return griglia


griglia = create_griglia(svc_prec, feat, img,chisquared_kernel)

griglia


mask = copy.deepcopy(img)
uniqueValues, occurCount = np.unique(gr, return_counts=True)
for val in uniqueValues:
    print(val)
    mask = copy.deepcopy(img)
    for i in range(griglia.shape[0]):
        for j in range(griglia.shape[1]):
            if(gr[i,j]==val):
                print(i,j)
                for ii in range(i*100,(i+1)*100):
                    for jj in range(j*100,(j+1)*100):
                        mask[ii,jj,:] = [0,0,0]
    plt.imshow(mask)
    plt.title("image prova3 class " + str(val))
    plt.savefig("grid_classification/results/test_masks/image prova3 class" + str(val) + ".jpg")





uniqueValues, occurCount = np.unique(gr, return_counts=True)
occurCount

occurCount


uniqueValues

plt.imshow(mask)

"""prova = copy.deepcopy(img)

for i in range(200):
    for j in range(200):
        prova[i,j,0] = 0
        prova[i,j,1] = 0
        prova[i,j,2] = 0


plt.imshow(prova)"""
