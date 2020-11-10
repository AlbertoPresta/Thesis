import pandas as pd
import matplotlib.pyplot as plt;
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical #to create dummy variable
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.svm import SVC
import warnings
import operator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel,chi2_kernel
warnings.filterwarnings('ignore')
image_size = 100
model = VGG16(weights='imagenet', include_top=False,input_shape = (image_size,image_size,3))

model.summary()

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
    y = []
    for ii,i in enumerate(lab):
        s = path+i+'/{}'
        temp = [s.format(i) for i in os.listdir(path+i+'/')]
        temp = temp
        for j in range(len(temp)):
            y.append(ii)
        x = x + temp
    x = np.asarray(x)
    y = np.asarray(y)
    if(shuf==True):
        x,y = unison_shuffled_copies(x,y)
    return x,y


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




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
        im = load_img(path, target_size=(100,100))
        if(cont==1):
            print(im.size)
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        im = imagenet_utils.preprocess_input(im)
        res.append(im)
    images = np.vstack(res)
    features = pre_model.predict(images, batch_size=64)
    features_flatten = features.reshape((features.shape[0], 3 * 3 * 512))
    return images, features, features_flatten






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
    print("------->",a)
    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num]==0 and hist_2[num]==0)])
    if(len(indexes)==0):
        return 0
    else:
        D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes]))
        return np.exp(- (D * 0.5) / a)





def compute_gram_matrix(data,kernel):
    samples,_ = np.shape(data)
    print(samples)
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
            acc.append(1)
        else:
            acc.append(0)

    print(len(acc))
    print(np.sum(acc))
    percentuale = np.sum(acc)/len(acc)

    return percentuale*100





lichens = os.listdir("grid_classification/patches/train")
x_train,y_train = list_of_path(lichens,"grid_classification/patches/train/")
x_test,y_test = list_of_path(lichens,"grid_classification/patches/test/")


%%time
images, features, features_flatten = feature_extraction(x_train,model)
test_images,feature_matrix_test,feature_array_test = feature_extraction(x_test,model)







svc_lin  = OneVsRestClassifier(SVC(kernel = 'linear',gamma = 'scale'),n_jobs = -1)
svc_lin = svc_lin.fit(features_flatten,y_train)







import joblib
joblib.dump(svc_lin, 'modelsvm.pk1')

# Step 2: `pool.apply` the `howmany_within_range()`
%%time
results = [pool.apply(compute_gram_matrix, args=(feat,chisquared_distance))]

len(results)
# Step 3: Don't forget to close
pool.close()

svc_lin.score(feature_array_test,y_test)*100
feat
%%time
#feat = np.array(features)
gram= compute_gram_matrix(features_flatten,chisquared_distance)
mean = np.mean(gram[np.triu_indices(np.shape(features_flatten[:2])[0])])
gram = np.exp(-(gram/mean)) # generalized Gaussian kernel
svc_prec  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
gram
svc_prec = svc_prec.fit(K,np.array(y_train))
svc_lin.score(feature_array_test,y_test)*100
#K = chi2_kernel(features_flatten, gamma=.5)



svc_prec = svc_prec.fit(gram,np.array(y_train))


svc_sig  = OneVsRestClassifier(SVC(kernel = 'sigmoid',gamma = 'scale'),n_jobs = -1)
svc_sig = svc_sig.fit(features_flatten,y_train)
svc_sig.score(feature_array_test,y_test)*100



svc_pol  = OneVsRestClassifier(SVC(kernel = 'poly',gamma = 'scale'),n_jobs = -1)
svc_pol = svc_pol.fit(features_flatten,y_train)
svc_pol.score(feature_array_test,y_test)*100
# define a function which extract feature for all my train images

#dense_sift_accuracy
np.save("grid_classification/results/features_flatten",features_flatten)


def predict_with_linear_SVM(features_array,feature_lab,feature_array_test,test_labels,num_classes = 28):
    """
    Function which exploits linear svm to predict classes of test data

    Input features_array = array with train features (flattened)
    Input feature_lab = labels of train features
    Input feature_array_test = array with test features (flattened)
    Input test_labels = labels of test features
    Input num_classes (15) = number of classes

    Output classif = array of svm classidicator
    Output prediction = array of prediction of test features
    Output acc = accuracy of prediction
    """

    classif = [SVC(kernel="linear") for _ in range(num_classes)] # array of classifier to be trained
    # one-vs-all approach with
    curr_label = 0
    for clf in classif:
        print(curr_label)
        v = np.array([1 if label==curr_label else 0 for label in feature_lab])
        clf = clf.fit(features_array, v)
        curr_label = curr_label + 1
    # now we want to test
    prediction = []
    print("prediction started")
    for image in feature_array_test:
        pred = np.array([np.dot(clf.coef_,image) + clf.intercept_ for clf in classif])
        prediction.append(np.argmax(pred))
    prediction = np.asarray(prediction)
    #calculate accuracy
    cont=0
    for i in range(len(prediction)):
        if prediction[i]==test_labels[i]:
            cont = cont +1
        else:
            continue

    acc = cont/len(prediction)

    return classif,prediction,acc







classif,prediction,acc = predict_with_linear_SVM(features_flatten, y_train, feature_array_test, y_test)
acc



acc = calculate_accuracy_nn(features_flatten, feature_array_test, y_train, y_test)



acc




kappas = np.arange(1,201,1)

t = []
for k in [1,2,3,4,5]:
    print(k)
    acc = calculate_accuracy_nn(features_flatten, feature_array_test, y_train, y_test,k = k)
    t.append(acc)

plt.plot(t)
np.max(t)





t












plt.figure(figsize=(15,15))
columns = ['svm-chi_squared','svm-lin','svm-rbf','svm-sig','svm-poly','nearest-neigh','knn']
rows = ['BGF','COL-BGF','DENSE-SIFT','SCALE-SIFT','VGG16-FEATURES']

cell_texts = [['22','20.8','21.3','6.4','12.9','24.53','25.46 (4)'],
              ['30.01','20.3','20.37','12.5','20.37','23.14','25.4 (4)'],
              ['14.35','11.57','11.57','12.03','3.7','12.5','12.5 (1)'],
            ['7.97','9.81','7.97','7.98','6.1','6.7','6.7 (1)'],
            ['-','43.51','39.58','41.2','42.47','18.17','18.17 (1)']]
# Add a table at the bottom of the axes
plt.box(on=None)
the_table = plt.table(cellText=cell_texts,rowLabels=rows,colLabels=columns,loc='center')
# Adjust layout to make room for the table:
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.subplots_adjust(left=0.2, bottom=0.2)
#plt.ylabel("Loss in ${0}'s".format(value_increment))
#plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
the_table.scale(1, 1.5)
plt.title('accuracy of processes')
# plt.show()





from sklearn import metrics
c = metrics.classification_report(y_test, prediction,target_names=lichens)

feat = np.load("grid_classification/features/train/ft/densesiftdescriptor_200.npy")
feat

X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
y = [0, 1, 0, 1]
K = chi2_kernel(features_flatten[:100], gamma=.5)
K
