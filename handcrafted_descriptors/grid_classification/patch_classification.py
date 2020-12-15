
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
import pickle

"""
extract training features and labels
"""

species[8]

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



def equal(a,b):
    for i in range(a.shape[0]):
        if a[i]!=b[i]:
            return False
    return True

for i,t in enumerate(tst_features):
    print(t.shape)
    if(equal(t,tst_features[i])==True):
        print(i)



def test_accuracy(test_features,features,kernel,mean, test_label,svc):
    num_objects = features.shape[0]
    res = []
    prediction = []
    for i,ft in enumerate(test_features):
        if(i%200 ==0):
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


def calculate_and_plot_precision_recall(tst_lab, pred, species, directory, string):
    precision, recall, fbeta, support = precision_recall_fscore_support(tst_lab, pred)
    df = pd.DataFrame({"X":species, "precision":precision,"recall":recall})
    df.plot(x="X", y=["precision", "recall"], kind="bar")
    plt.tight_layout()
    plt.savefig(directory +'precision_recall_class4class'+string+'.jpg')



acc = []
prec = []
rec = []



trf = '../features_for_python/features_pf_450_8_rbf/training_features.mat'
trlab = '../features_for_python/features_pf_450_8_rbf/training_lab_features.mat'

tstf = '../features_for_python/features_pf_400_8_rbf_classic_sift/testing_features.mat'
tstlab = '../features_for_python/features_pf_400_8_rbf_classic_sift/testing_lab_features.mat'

vldf = '../features_for_python/features_pf_400_8_rbf/validate_features.mat'
cldlab = '../features_for_python/features_pf_400_8_rbf/validate_lab_features.mat'

training_feat,tr_lab = load_descriptor_from_matfile(trf, trlab)
tst_features, tst_lab = load_descriptor_from_matfile(tstf,tstlab)
validate_features, validate_lab = load_descriptor_from_matfile(vldf,cldlab)
tr_lab = tr_lab -1
tst_lab = tst_lab -1
validate_lab = validate_lab - 1

svm, mean, gram = define_and_train_svm(training_feat,tr_lab,'rbf',distance = None)
linear_score = svm.score(validate_features,validate_lab)

pred = svm.predict(tst_features)

#t,pred = test_accuracy(tst_features, training_feat, krn.chisquared_kernel, mean, tst_lab ,svm)

acc.append(t)
print('accuracy: ',linear_score)

label = np.arange(0,20)

#linear_score = svm.score(tst_features,tst_lab)
#pred = svm.predict(tst_features)
df = utils.evaluated_prediction(pred, tst_lab, species)
cm = utils.build_confusion_matrix(df, pred, tst_lab,species)
fig=plt.figure(figsize=(30, 15))
utils.plot_confusion_matrix(cm,species,"Phow_descriptors_8_chisquared_"+c+'_',"grid_classification/results/confusion_matrix/",normalize=False,title='Confusion matrix')
precision, recall, fbeta, support = precision_recall_fscore_support(tst_lab, pred)

for c in ['400']:


    trf = '../features_for_python/features_pf_' + c + '_8_rbf/training_features.mat'
    trlab = '../features_for_python/features_pf_'+ c + '_8_rbf/training_lab_features.mat'

    tstf = '../features_for_python/features_pf_' + c + '_8_rbf/testing_features.mat'
    tstlab = '../features_for_python/features_pf_' + c + '_8_rbf/testing_lab_features.mat'

    vldf = '../features_for_python/features_pf_400_8_rbf/validate_features.mat'
    vldlab = '../features_for_python/features_pf_400_8_rbf/validate_lab_features.mat'

    training_feat,tr_lab = load_descriptor_from_matfile(trf, trlab)
    tst_features, tst_lab = load_descriptor_from_matfile(tstf,tstlab)
    validate_features, validate_lab = load_descriptor_from_matfile(vldf,cldlab)
    tr_lab = tr_lab -1
    tst_lab = tst_lab -1
    validate_lab = validate_lab - 1

    svm, mean, gram = define_and_train_svm(training_feat,tr_lab,'precomputed',distance = krn.chisquared_distance)
    t,pred = test_accuracy(tst_features, training_feat, krn.chisquared_kernel, mean, tst_lab ,svm)
    tt,_ = test_accuracy(validate_features, training_feat, krn.chisquared_kernel, mean, validate_lab ,svm)
    print('-------> ', tt, '-----------------')
    filename = 'grid_classification/model/finalized_model.sav'
    pickle.dump(svm, open(filename, 'wb'))
    pickle.dump(mean,open('grid_classification/model/mean.sav','wb'))

    pickle.dump(training_feat, open('grid_classification/model/training.sav', 'wb'))
    pickle.dump(mean,open('grid_classification/model/mean.sav','wb'))
    acc.append(t)
    print('accuracy: ',t)

    label = np.arange(0,20)

    #linear_score = svm.score(tst_features,tst_lab)
    #pred = svm.predict(tst_features)
    df = utils.evaluated_prediction(pred, tst_lab, species)
    cm = utils.build_confusion_matrix(df, pred, tst_lab,species)
    fig=plt.figure(figsize=(30, 15))
    utils.plot_confusion_matrix(cm,species,"Phow_descriptors_8_chisquared_"+c+'_',"grid_classification/results/confusion_matrix/",normalize=True,title='Confusion matrix')
    precision, recall, fbeta, support = precision_recall_fscore_support(tst_lab, pred)
    prec.append(precision)
    rec.append(recall)


p = np.array(prec)
np.mean(p,axis = 1).shape
precision_mean = np.mean(p,axis = 1)
rec_mean = np.mean(p,axis = 1)
os.listdir()
precision_mean


plt.figure(figsize=(8, 5))
#plt.plot(['250','300','350','400','450','500','550','600','650','700','750','800','850','900'],acc, label = 'accuracy',color = 'blue')
#plt.plot(['250','300','350','400','450','500','550','600','650','700','750','800','850','900'],precision_mean, label = 'precision',color = 'green')
plt.plot(['250','300','350','400','450','500','550','600','650','700','750','800','850','900'],rec_mean, label = 'recall',color = 'blue')
plt.ylabel('recall in percentage')
plt.xlabel('vocabulary size')
plt.title('How recall varies when I change size of vocabulary (SIFT descriptor with chi_squared SVM)')
plt.legend()
plt.grid()
plt.savefig('grid_classification/results/recall_varying_vocsize.jpg')

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




I = cv2.imread('grid_classification/descriptors/img/text_image_N1.jpg')
I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)

plt.imshow(I)

true_classification = np.zeros((10,10))

true_classification[0:5,0:5] = 8
true_classification[5:10,0:5] = 4
true_classification[0:5,5:10] = 17
true_classification[5:10,5:10] = 13
true_classification


c = np.load('grid_classification/descriptors/labels/text_image_N1.npy')
c

extract_accuracy_on_test_image(I,'grid_classification/descriptors/img/text_image_N1.npy', svm, true_classification,  training_feat,mean,krn.chisquared_kernel)
