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


c.shape

c = np.load('validation_collages.npy')


t = np.zeros([256,256,3])

t[:,:,0] = c[0,:,:,0]
t[:,:,1] = c[0,:,:,1]
t[:,:,2] = c[0,:,:,2]
plt.imshow(c[0,:,:,1])


CENTERS = [(300,300),(300,700),(700,300),(700,700)]#CENTERS = [(200,200),(200,400),(400,200),(400,200),(600,600),(600,800),(800,600),(800,800)]
SIZE = 200
pth = "../final_dataset/train"
test_pth = "../final_dataset/test"
lichens = os.listdir(pth)
#feat = np.load("grid_classification/features/train/ft/color_RGB_200.npy")
#label = np.load("grid_classification/features/train/lab/color_RGB_200_lab.npy")

#test_feat = np.load("grid_classification/features/test/ft/color_RGB_200.npy")
#test_label = np.load("grid_classification/features/test/lab/color_RGB_200_lab.npy")

def scale_sift_desc(pth):
    sift = cv2.xfeatures2d.SIFT_create()
    lichens = os.listdir(pth)
    feat = []
    label = []
    l = -1
    for lichen in lichens:
        print("--------- ",lichen," -----------")
        l = l+1
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
                temporary = []
                crop_img = im[crd[0]-SIZE:crd[0] +SIZE,crd[1] - SIZE:crd[1] +SIZE,:]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                points,features = sift.detectAndCompute(crop_img,None)
                for ft in features:
                    feat.append(ft.reshape(-1))
                    label.append(l)
    return feat, label



feat, label =scale_sift_desc(pth)

feat = np.array(feat)
label = np.array(label)






from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=300, random_state=0).fit(feat)


def leg_col_co_descriptors(pth,level = 2, bins = 16, distance = [1], direction = [0, np.pi/2, 3*np.pi/2]):
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
            im = cv2.resize(im,(1000,1000))
            im = im[100:900,100:900,:]
            for ii,x in enumerate(x_center):
                for jj,y in enumerate(y_center):
                    pt = (x,y)
                    image = im[x- SIZE:x + SIZE,y - SIZE:y + SIZE,:]
                    legendre = Legendre_vector(image,level)
                    histograms = color_hist_vector(image,  bins = bins)
                    co_occurrence = co_occurrence_vector(image,distance = distance, direction = direction)
                    temp =np.concatenate([legendre,histograms,co_occurrence])

                    features.append(temp)
                    label.append(l)
        l = l + 1
    return features, label, dic_label





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

    return features, label , dic_label, container_features


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
            #imR = im[:,:,0]
            #imG = im[:,:,1]
            #imB = im[:,:,2]

            for ii,x in enumerate(x_center):
                for jj,y in enumerate(y_center):
                    pt = (x,y)

                    desc = bgf.BGF(im,x,y,SIZE)
                    print("-----> ",desc.shape)
                    #desc_G = bgf.BGF(imG,x,y,SIZE)
                    #desc_B = bgf.BGF(imB,x,y,SIZE)

                    imm = im[x-SIZE:x+SIZE,y-SIZE:y+SIZE,:]
                    #color = color_hist_vector(imm,  bins = 16)
                    color = np.asarray(color)
                    #color = color.reshape(-1)
                    desc = np.asarray(desc)
                    desc = desc.reshape(-1)

                    #desc_G = np.asarray(desc_G)
                    #desc_G = desc_G.reshape(-1)

                    #desc_B = np.asarray(desc_B)
                    #desc_B = desc_B.reshape(-1)

                    temp = np.concatenate([desc,color])

                    features.append(temp)
                    label.append(l)
        l = l + 1
    return features, label, dic_label




def surf_descriptors(pth,thresh = 400):
    surf = cv2.xfeatures2d.SURF_create(thresh)
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
                print("store")
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
                print(type(crop_img))
                _,feaArrSingle_R = surf.compute(crop_img[:,:,0],[kp])
                _,feaArrSingle_G = surf.compute(crop_img[:,:,1],[kp])
                _,feaArrSingle_B = surf.compute(crop_img[:,:,2],[kp])

                print(type(feaArrSingle_R))
                print(type(feaArrSingle_G))
                print(type(feaArrSingle_B))
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

    return features, label , dic_label, container_features












def chisquared_distance(hist_1, hist_2):

    k = hist_1.shape[0]

    indexes = np.array([num for num in range(k) if not (hist_1[num] == 0 and hist_2[num] == 0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes])) # the Chi-squared distance
        # plug into the generalized Gaussian kernel
        return 0.5 * D




def chisquared_kernel(hist_1,hist_2,a):

    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num]==0 and hist_2[num]==0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes]))
        return np.exp(- (D * 0.5) / a)








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









def test_accuracy(test_features,features,kernel,mean, test_label,svc):
    num_objects = features.shape[0]
    print(num_objects)
    res = []
    prediction = []
    for i,ft in enumerate(test_features):
        #print(ft)

        pred = svc.predict(np.array([kernel(ft, features[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
        prediction.append(pred[0])
        #print(i,": ",pred," : ",test_label[i])
        if(pred==test_label[i]):
            res.append(1)
        else:
            res.append(0)

    res = np.array(res).reshape(-1)
    return np.sum(res)/res.shape, np.array(prediction)









features, label , dic_label= RGB_descriptors(pth)
feat = np.array(features)
label = np.array(label)
#np.save("grid_classification/features/train/ft/dic_label.npy",dic_label)
np.save("grid_classification/features/train/ft/scalesiftdescriptor_200.npy",feat)
np.save("grid_classification/features/train/lab/scalesiftdescriptor_200_lab.npy",label)


test_features, test_label , dic_label_test = RGB_descriptors(test_pth)
test_feat = np.array(test_features)
test_label = np.array(test_label)


%%time
#feat = np.array(features)
gram= compute_gram_matrix(feat,chisquared_distance)
mean = np.mean(gram[np.triu_indices(np.shape(feat)[0])])
c = np.mean(gram)
gram = np.exp(-(gram/mean)) # generalized Gaussian kernel
svc_prec  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
svc_prec = svc_prec.fit(gram,np.array(label))
#pickle.dump(svc_prec, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_precomputed', 'wb'))


svc_rfb  = OneVsRestClassifier(SVC(kernel = 'rbf',gamma = 'scale'),n_jobs = -1)
svc_rfb = svc_rfb.fit(feat,label)


pickle.dump(svc_rfb, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_guassian', 'wb'))

gram[0]
from sklearn.metrics.pairwise import additive_chi2_kernel,chi2_kernel

r = chi2_kernel(feat,gamma = 0.5)
r
gram

svc_lin  = OneVsRestClassifier(SVC(kernel = 'linear',gamma = 'scale'),n_jobs = -1)
svc_lin = svc_lin.fit(feat,label)


pickle.dump(svc_lin, open('grid_classification/features/train/models/scale_dense_sift_descriptors_200_linear', 'wb'))

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
