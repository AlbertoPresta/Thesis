import os
import cv2
from grid_classification import dense_sift
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

"""
RESIZE ALL IMAGE TO 1000*1000
TAKE CENTRAL CROP OF 800*800
TAKE 4 SUB IMAGE OF 400*400--->[100:500,100_500],[100:500,500:900],[500:900,100_500],[500:900,500:900]
FOR EACH OF THIS SUBIMAGES, CALCULATE DENSE SIFT descriptors
K-MEANS CLUSTERING (MAYBE)
64*4 VECTORS
"""
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

t = cof.color_hist_vector(cv2.imread("../final_dataset/test/Amandinea_punctata/texture04.jpg"),  bins = 36)

x_center = np.arange(16,800,16)

x_center

extr = dense_sift.SingleSiftExtractor(200)

#test_extr = dense_sift.SingleSiftExtractor(50)



def calculate_dataset(pth,extractor = extr):
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


features, label , dic_label, container_features = calculate_dataset(pth)




test_features, test_label , dic_label_test , total= calculate_dataset(test_pth)





"""
SUPPORT VECTOR MACHINE
"""









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



d = chisquared_kernel(features[0],features[1],1)

feat = np.array(features)


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



%%time
f = compute_gram_matrix(feat,chisquared_distance)
gram = f
mean = np.mean(gram[np.triu_indices(np.shape(features)[0])])
gram = np.exp(-(gram/mean)) # generalized Gaussian kernel
svc  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
svc = svc.fit(gram,np.array(label))




def test_accuracy(test_features,features,kernel,mean, test_label):
    num_objects = features.shape[0]
    print(num_objects)
    res = []
    for i,ft in enumerate(test_features):
        #print(ft)
        pred = svc.predict(np.array([kernel(ft, features[num, :], mean) for num in range(num_objects)]).reshape(1, -1))
        #print(i,": ",pred," : ",test_label[i])
        if(pred==test_label[i]):
            res.append(1)
        else:
            res.append(0)

    res = np.array(res).reshape(-1)
    return np.sum(res)/res.shape


acc = test_accuracy(test_features, feat,chisquared_kernel,mean,test_label)
acc


"""
FACCIAMO UNA PROVA SU UN IMMAGINE DI TEST
"""

immagine = Image.open("../prova2.jpg")
immagine = asarray(immagine)
plt.imshow(immagine)
plt.imshow(immagine[800:1000,800:1200,:])
immagine = asarray(immagine)


immagine = cv2.resize(immagine, (2000,2000))

x_center = np.arange(100,2000,200)
y_center = np.arange(100,2000,200)
SIZE = 100
res = np.zeros((10,10))
sift = cv2.xfeatures2d.SIFT_create()
for i,x in enumerate(x_center):
    for j,y in enumerate(y_center):
        crop_img = immagine[x-SIZE:SIZE +x,y - SIZE:y +SIZE,:]
        kp = cv2.KeyPoint(100,100,SIZE)

        _,feaArrSingle_R = sift.compute(crop_img[:,:,0],[kp])
        _,feaArrSingle_G = sift.compute(crop_img[:,:,1],[kp])
        _,feaArrSingle_B = sift.compute(crop_img[:,:,2],[kp])

        feaArrSingle_R =  feaArrSingle_R.reshape(-1)
        feaArrSingle_G =  feaArrSingle_G.reshape(-1)
        feaArrSingle_B =  feaArrSingle_B.reshape(-1)

        temp = np.concatenate([feaArrSingle_R,feaArrSingle_G,feaArrSingle_B])

        pred = svc.predict(np.array([chisquared_kernel(temp, feat[num, :], mean) for num in range(feat.shape[0])]).reshape(1, -1))
        res[i,j] = pred[0]


res



"""
VISUALIZE IMAGE PATCH
"""

def visualize_image_patches(images,lichen_name):
    lista = images[lichen_name][0]
    print(images[lichen_name][ 1])

    fig=plt.figure(figsize=(100, 100))
    columns = len(lista)//4
    rows = len(lista) - columns
    for i in range(len(lista)):
        img = lista[i]
        fig.add_subplot(rows, columns, i +1)
        plt.imshow(img)
    plt.show()


visualize_image_patches(images,'Chrysothrix_candelaris')







"""
bag of word kmeans
"""


def get_visual_words_for_kmeans(descriptors,k):
    kmeans = KMeans(n_clusters=k, n_init=5, n_jobs=-1)
    kmeans = kmeans.fit(descriptors)
    return kmeans

bovw = get_visual_words_for_kmeans(features,10)


container_features['Gyalolechia_flavorubescens']




def compute_histogram_for_training_set(container_features,vocabulary,lichens_list = lichens):
    lichens_bag_of_words = {}
    k = len(vocabulary)
    for lichene in lichens_list:
        print("lichene: ",lichene)
        features = container_features[lichene][1]
        numero_texture = len(features)//4
        print("ppppp",numero_texture)
        lichen_histograms = np.zeros((numero_texture,10))
        for i in range(numero_texture):
            words = []
            for j in range(4):
                arrays = features["texture0" + str(i+1) + "_crop_ " + str(j)]
                prova,_ = vq(arrays, bovw.cluster_centers_, check_finite=False)
                prova = prova.reshape(-1)
                print(prova)
                words = words + list(prova)

            print(words)
            lichen_histograms[i],_ = np.histogram(words, bins = np.arange(k+1))
            print("--->",lichen_histograms[i])


            lichens_bag_of_words[lichene + "texture0" + str(i+1) + "_crop_ " + str(j) ] = lichen_histograms[i]
    return lichens_bag_of_words






lichens_bag_of_words = compute_histogram_for_training_set(container_features,bovw.cluster_centers_)






# creo la lista di traning set con le rispettive label:




def create_training_set(lichen_bow, lich_dic):
    res = []
    lab = []
    lichens_all = list(lichen_bow.keys())
    for lichen in lichens_all:
        print(lichen)
        training_vectors = lichen_bow[lichen]
        print(training_vectors.shape)
        res.append(training_vectors.reshape(-1))
        lab.append(lich_dic[lichen.split("texture")[0]])
    res = np.array(res)
    lab = np.array(lab)
    return res,lab





train_data,train_lab = create_training_set(lichens_bag_of_words,dic_label)

train_lab











"""
PROVIAMO DIRETTAMENTO CON NON LINEAR SUPPORT VECTOR MACHINE

"""



def chisquared_distance(hist_1, hist_2):

    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num] == 0 and hist_2[num] == 0)])
    D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes])) # the Chi-squared distance
    # plug into the generalized Gaussian kernel
    return 0.5 * D




def chisquared_kernel(hist_1,hist_2,a):

    k = hist_1.shape[0]
    indexes = np.array([num for num in range(k) if not (hist_1[num]==0 and hist_2[num]==0)])
    D = np.sum(np.square(hist_1[indexes] - hist_2[indexes]) / (hist_1[indexes] + hist_2[indexes]))
    return np.exp(- (D * 0.5) / a)







def compute_gram_matrix(data,kernel):
    samples,_ = np.shape(data)
    matrix = np.zeros((samples,samples))
    for r in range(samples):
        for c in range(r,samples):
            tmp = kernel(data[r],data[c])
            matrix[r,c] = tmp
            matrix[c,r] = tmp
    return matrix


%%time
f = compute_gram_matrix(train_data,chisquared_distance)


def chi_square_svm(train_data_histograms,vocabulary,labels,classes,distance):
    """
    fit a chi square support vector machine
    """
    # number of visual words to use
    k = len(vocabulary)

    print("start calculating Gram matrix")
    #pre-compute the gram matrix among training instances
    gram =  compute_gram_matrix(train_data_histograms,distance)
    mean = np.mean(gram[np.triu_indices(np.shape(train_data_histograms)[0])])
    gram = np.exp(-(gram/mean)) # generalized Gaussian kernel


    num_classes = len(classes)
    print("start training svm")
    svc  = OneVsRestClassifier(SVC(kernel = 'precomputed'),n_jobs = -1)
    svc = svc.fit(gram,np.array(labels))

    return svc, gram,mean


%%time
svcc , gram, mean = chi_square_svm(train_data, bovw.cluster_centers_, train_lab, list(dic_label.keys()), chisquared_distance)




def test_accuracy(svcc, train_data,train_label,test_path = "../final_dataset/test")
train_data[0]

immagine = Image.open("../final_dataset/test/Amandinea_punctata/texture04.jpg")
im = asarray(immagine)
im = cv2.resize(im,(1000,1000))
plt.imshow(im)


t = []
for ii,crd in enumerate(CENTERS):
    crop_img = im[crd[0]-SIZE:crd[0] +SIZE,crd[1] - SIZE:crd[1] +SIZE,:]
    kp = cv2.KeyPoint(crd[0],crd[1],SIZE)

    _,feaArrSingle_R = sift.compute(crop_img[:,:,0],[kp])
    _,feaArrSingle_G = sift.compute(crop_img[:,:,1],[kp])
    _,feaArrSingle_B = sift.compute(crop_img[:,:,2],[kp])

    feaArrSingle_R =  feaArrSingle_R.reshape(-1)
    feaArrSingle_G =  feaArrSingle_G.reshape(-1)
    feaArrSingle_B =  feaArrSingle_B.reshape(-1)


    temp = np.concatenate([feaArrSingle_R,feaArrSingle_G,feaArrSingle_B])
    t.append(temp)

r = []
t = np.array(t)

g,_ = vq(t, bovw.cluster_centers_, check_finite=False)

prova,_ = np.histogram(g, bins = np.arange(10+1))

svcc.predict(np.array([chisquared_kernel(prova,train_data[num], mean) for num in range(len(train_data))]).reshape(1, -1))






dic_label
