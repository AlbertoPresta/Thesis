import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from descriptor import surf as srf
from descriptor import binary_gabor_features as bgf
%matplotlib inline

"""
ORGANIZATION OF THE IMAGES: save them in a dictionary, label by label

"""



def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        print("------> ",filename)
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            print(cat)
            if('.DS_Store' in cat):
                continue
            img = cv2.imread(path + "/" + cat)
            img = cv2.resize(img,(1000,1000))
            img = img[300:700,300:700]
            plt.imshow(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images


images.keys()




images = load_images_from_folder("../final_dataset/train")
query = load_images_from_folder("../final_dataset/test")


"""
visualize images

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

fig = plt.figure(figsize=(50, 50))
grid = ImageGrid(fig, 111, nrows_ncols=(len(images), 3),axes_pad=0.1, )

for ax, tp in zip(grid,images):
    # Iterating over the grid returns the Axes.
    for j,img in enumerate(images[tp]):
        if(j>=3):
            continue
        ax.imshow(img)

plt.show()

plt.imshow(images[tp][])



"""
descriptor creation ----> prendere SIFT+ color + mr8 + legendre + bgf
"""
def sift_features(images):

    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        print(key)
        features = []
        for img in value:
            print(type(img))
            #des = srf.create_surf_descriptors(img)
            kp, des = sift.detectAndCompute(img,None)
            print(type(des))

            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]




srf.create_surf_descriptors(cv2.imread("../prova.jpg"))
sifts = sift_features(images)
descriptor_list = sifts[0]
# Takes the sift features that is seperated class by class for train data
all_bovw_feature = sifts[1]

test_sifts = sift_features(query)
test_bovw_feature = test_sifts[1]


"""
k-means clustering!
k = 10
"""

def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words

# Takes the central points which is visual words
visual_words = kmeans(110, descriptor_list)

"""
creazione degli istogrammi per il training set
"""
from cyvlfeat.sift import phow


"""
immagine di test---> prendo griglia e cella grande 200*200
divido in 16 quadratini e calcolo descrittori e poi istogramma
---> knn

"""
# Find the index of the closest central point to the each sift descriptor.
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind
# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class.
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        print(key)
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

# Creates histograms for train data
bovw_train = image_class(all_bovw_feature, visual_words)
# Creates histograms for test data
bovw_test = image_class(test_bovw_feature, visual_words)

# 1-NN algorithm. We use this for predict the class of test images.
# Takes 2 parameters. images is the feature vectors of train images and tests is the feature vectors of test images
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
def knn(images, tests):
    num_test = 0
    correct_predict = 0
    class_based = {}

    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0] # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a" #predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        #minimum = L1_dist(tst,train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        #dist = L1_dist(tst,train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key

            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]

# Call the knn function
results_bowl = knn(bovw_train, bovw_test)

results_bowl

# Calculates the average accuracy and class based accuracies.
def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))

# Calculates the accuracies and write the results to the console.
accuracy(results_bowl)
