
# Import python library for this notebook
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # package for plot function
from scipy.io import loadmat
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from grid_classification import kernels as krn
from grid_classification import accuracy_and_test as aat
from grid_classification import utils
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

"""
extract training features and labels
"""



species = ['Arthonia_radiata','Caloplaca_cerina','Candelariella_reflexa','Candelariella_xanthostigma','Chrysothrix_candelaris','Flavoparmelia_caperata','Gyalolechia_flavorubescens','Hyperphyscia_adglutinata'
        ,'Lecanora_argentata','Lecanora_chlarotera','Lecidella_elaeochroma','Melanelixia_glabratula'
        ,'Phaeophyscia_orbicularis','Physcia_biziana','Physconia_grisea','Ramalina_farinacea','Ramalina_fastigiata','Xanthomendoza_fallax','Xanthomendoza_fulva','flavoparmenia_soredians']

def load_descriptor_from_matfile(ft_path,lab_path):
    feat = loadmat(ft_path)
    feat = feat['features']
    lab = loadmat(lab_path)
    lab = lab['labels']
    return feat, lab


def define_and_train_svm(tr_feature, tr_lab, kernel_type, distance = None):
    if distance == None:
        print('here')
        svc  = OneVsRestClassifier(SVC(kernel = kernel_type,gamma = 'scale'),n_jobs = -1)
        svc = svc.fit(tr_feature, tr_lab)
        return svc, 0, np.zeros(tr_feature.shape[0])
    else:
        gram= krn.compute_gram_matrix(tr_feature,distance)
        mean = np.mean(gram[np.triu_indices(np.shape(tr_feature)[0])])
        gram = np.exp(-(gram/mean)) # generalized Gaussian kernel
        svc  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
        svc = svc.fit(gram,np.array(tr_lab))
        return svc, mean,gram




def test_accuracy(test_features,features,kernel,mean, test_label,svc):
    num_objects = features.shape[0]
    print(num_objects)
    print(test_features.shape[0])
    res = []
    prediction = []
    for i,ft in enumerate(test_features):
        print(i)

        pred = svc.predict(np.array([kernel(ft, features[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
        prediction.append(pred[0])
        #print(i,": ",pred," : ",test_label[i])
        if(pred==test_label[i]):
            res.append(1)
        else:
            res.append(0)

    res = np.array(res).reshape(-1)
    return np.sum(res)/res.shape, np.array(prediction)













training_feat,tr_lab = load_descriptor_from_matfile('grid_classification/python_data/training_features.mat', 'grid_classification/python_data/training_lab.mat')
tst_features, tst_lab = load_descriptor_from_matfile('grid_classification/python_data/test_features.mat','grid_classification/python_data/test_lab.mat')
tr_lab = tr_lab -1
tst_lab = tst_lab -1


svm, mean, gram = define_and_train_svm(training_feat,tr_lab,'precomputed',distance = krn.chisquared_distance)
t,pred = test_accuracy(tst_features, training_feat, krn.chisquared_kernel, mean, tst_lab ,svm)


t

label = np.arange(0,20)

linear_score = svc.score(test_feat,test_label)
#pred = svm.predict(tst_features)
df = utils.evaluated_prediction(pred, tst_lab, species)
cm = utils.build_confusion_matrix(df, pred, tst_lab,species)
fig=plt.figure(figsize=(30, 15))
utils.plot_confusion_matrix(cm,species,"Phow_descriptors_8grid_chi_squared_kernel","grid_classification/results/confusion_matrix/",normalize=True,title='Confusion matrix')







def calculate_and_plot_precision_recall(tst_lab, pred, species, directory, string):
    precision, recall, fbeta, support = precision_recall_fscore_support(tst_lab, pred)
    df = pd.DataFrame({"X":species, "precision":precision,"recall":recall})
    df.plot(x="X", y=["precision", "recall"], kind="bar")
    plt.tight_layout()
    plt.savefig(directory +'precision_recall_class4class'+string+'.jpg')


fig=plt.figure(figsize=(30, 15))
calculate_and_plot_precision_recall(tst_lab,pred,species,'grid_classification/results/precision_recall/','phow_feat_8grdi_chi_squared')




"""

§§§§§§§§§

"""
def extract_accuracy_on_test_image(I,descriptor_pth, classifier, true_classification, features, mean ,kernel ):

    descriptors = loadmat(descriptor_pth)['descriptors']
    (m,n,visual_words) = descriptors.shape
    cl = np.zeros((10,10))
    # applichiamo il classficatore svm suicrops
    for i in range(m):
        for j in range(n):
            dsc = descriptors[i,j,:]
            cl[i,j] = classifier.predict(np.array([kernel(dsc, features[num, :], mean) for num in range(features.shape[0])]).reshape(1, -1))


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
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100,(j+1)*100),d[l1],-1)
            elif cl[i,j] == l2:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100,(j+1)*100),d[l2],-1)
            elif  cl[i,j] == l3:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100,(j+1)*100),d[l3],-1)
            elif cl[i,j] == l4:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100,(j+1)*100),d[l4],-1)
            else:
                cv2.rectangle(mask_I, (i*100,j*100),((i+1)*100,(j+1)*100),[125,125,125],-1)


    fig=plt.figure(figsize=(30, 15))
    f, axarr = plt.subplots(1,3)
    print(axarr.shape)
    axarr[0] = plt.imshow(I)
    axarr[1] =plt.imshow(mask.astype(np.uint8))
    axarr[2] = plt.imshow(mask_I.astype(np.uint8 ))


plt.ims

plt.imshow(I)

I = cv2.imread('grid_classification/text_image_N1.jpg')
I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)

true_classification = np.zeros((10,10))

true_classification[0:5,0:5] = 6
true_classification[5:10,0:5] = 19
true_classification[0:5,5:10] = 3
true_classification[5:10,5:10] = 8
true_classification


extract_accuracy_on_test_image(I,'grid_classification/descriptors/dcs/text_image_N1.mat', svm, true_classification,  training_feat,mean,krn.chisquared_kernel)
