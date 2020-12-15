import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC

c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])




species = ['Arthonia_radiata','Caloplaca_cerina','Candelariella_reflexa','Candelariella_xanthostigma','Chrysothrix_candelaris','Flavoparmelia_caperata','Gyalolechia_flavorubescens','Hyperphyscia_adglutinata'
        ,'Lecanora_argentata','Lecanora_chlarotera','Lecidella_elaeochroma','Melanelixia_glabratula'
        ,'Phaeophyscia_orbicularis','Physcia_biziana','Physconia_grisea','Ramalina_farinacea','Ramalina_fastigiata','Xanthomendoza_fallax','Xanthomendoza_fulva','flavoparmenia_soredians']

def load_descriptor_from_matfile(ft_path,lab_path):
    feat = loadmat(ft_path)
    feat = feat['dsc']
    lab = loadmat(lab_path)
    lab = lab['lab']
    return feat, lab



def gabor_kernel(u,v,vec):

    m = u.shape[0]
    res = 0
    for i in range(0,m,2):
        mu_u = u[i]
        mu_v = v[i]

        sig_u = u[i+1]
        sig_v = v[i+1]

        mu_sum = np.abs((mu_u - mu_v)/vec[i])
        sig_sum = np.abs((sig_u - sig_v)/vec[i+1])

        res += (mu_sum + sig_sum)
    res = np.exp(-res)
    return res



def gabor_distance(u,v,vec):
    m = u.shape[0]
    res = 0
    for i in range(0,m,2):
        mu_u = u[i]
        mu_v = v[i]

        sig_u = u[i+1]
        sig_v = v[i+1]

        mu_sum = np.abs((mu_u - mu_v)/vec[i])
        sig_sum = np.abs((sig_u - sig_v)/vec[i+1])

        res += (mu_sum + sig_sum)
    return res




def compute_gram_matrix(data,kernel):
    samples,_ = np.shape(data)
    print(samples)
    vec= np.std(data,axis = 0)
    matrix = np.zeros((samples,samples))
    for r in range(samples):
        print(r)
        for c in range(r,samples):
            tmp = kernel(data[r],data[c],vec)
            matrix[r,c] = tmp
            matrix[c,r] = tmp
    return matrix


def define_and_train_svm(tr_feature, tr_lab, kernel_type, distance = None):
    if distance == None:
        print('here')
        svc  = OneVsRestClassifier(SVC(kernel = kernel_type,gamma = 'scale'),n_jobs = -1)
        svc = svc.fit(tr_feature, tr_lab)
        return svc, 0, np.zeros(tr_feature.shape[0])
    else:
        gram=  compute_gram_matrix(tr_feature,distance)
        #mean = np.mean(gram[np.triu_indices(np.shape(tr_feature)[0])])
        gram = np.exp(-(gram)) # generalized Gaussian kernel
        svc  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
        svc = svc.fit(gram,np.array(tr_lab))
        return svc, mean,gram

def test_accuracy(test_features,features,kernel,vec, test_label,svc):
    num_objects = features.shape[0]
    print(num_objects)
    print(test_features.shape[0])
    res = []
    prediction = []
    for i,ft in enumerate(test_features):
        print(i)

        pred = svc.predict(np.array([kernel(ft, features[num, :], vec) for num in range(num_objects)]).reshape(1, -1))
        prediction.append(pred[0])
        #print(i,": ",pred," : ",test_label[i])
        if(pred==test_label[i]):
            res.append(1)
        else:
            res.append(0)

    res = np.array(res).reshape(-1)
    return np.sum(res)/res.shape, np.array(prediction)



training_feat,tr_lab = load_descriptor_from_matfile('gabor_classification/dsc/training_descriptors.mat', 'gabor_classification/dsc/training_labels.mat')
tst_features, tst_lab = load_descriptor_from_matfile('gabor_classification/dsc/test_descriptors.mat','gabor_classification/dsc/test_labels.mat')
tr_lab = tr_lab -1
tst_lab = tst_lab -1



svm.score(np.array(tst_features),np.array(tst_lab))

svm, mean, gram = define_and_train_svm(training_feat,tr_lab,'precomputed',distance = gabor_distance)
vec = np.std(training_feat,axis = 0)
t,pred = test_accuracy(tst_features, training_feat, gabor_kernel, vec, tst_lab ,svm)


t

label = np.arange(0,20)

linear_score = svc.score(test_feat,test_label)
#pred = svm.predict(tst_features)
df = utils.evaluated_prediction(pred, tst_lab, species)
cm = utils.build_confusion_matrix(df, pred, tst_lab,species)
fig=plt.figure(figsize=(30, 15))
utils.plot_confusion_matrix(cm,species,"Phow_descriptors_8grid_chi_squared_kernel","grid_classification/results/confusion_matrix/",normalize=True,title='Confusion matrix')
