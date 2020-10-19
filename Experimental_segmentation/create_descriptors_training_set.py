import numpy as np
import os
from descriptor import binary_gabor_features as bgf
from descriptor import distance as dst
from descriptor import co_occurrence_feature
from descriptor import surf
import cv2
import matplotlib.pyplot as plt





def calculate_BGF_training_vectors_and_centroids(data_path = "../final_dataset/"):
    # elencare tutti i tipi diversi di lichene che ho
    lichens_species = os.listdir(data_path)

    centroids = []
    lichens_name = []
    all_training_vectors = []
    names = []
    #per ogni specie, calcolo i singoli BGF e il centroid
    for l,lichens in enumerate(lichens_species):
        names.append(lichens)
        print(lichens)
        pth = os.path.join(data_path,lichens)
        lichens_images = [pth + '/'+i for i in os.listdir(pth)]
        lichens_vec = np.zeros((len(lichens_images),55296))
        print("lichens_vec",lichens_vec.shape)
        for ii,x in enumerate(lichens_images):
            print(x)
            res = bgf.calculate_BGF_mean(x, 50)

            lichens_vec[ii,:] =  res

        all_training_vectors.append(lichens_vec)

        lichen_centroid = np.zeros(55296)

        for i in range(lichens_vec.shape[1]):
            temp = np.zeros(lichens_vec.shape[0])
            for jj in range(lichens_vec.shape[0]):
                temp[jj] = lichens_vec[jj,ii]
            lichen_centroid[ii] = np.mean(temp)

        centroids.append(lichen_centroid)

    np.save('../files/lichens__BGF_centroid',centroids)
    np.save('../files/lichens_BGF_vectors',all_training_vectors)
    np.save('../lichens_name',names)
    return centroids, all_training_vectors







"""
************************************* SURF DESCRIPTOR **********************
"""








def calculate_surf_vectors_and_centroids(data_path = "../final_dataset/"):
    # elencare tutti i tipi diversi di lichene che ho
    lichens_species = os.listdir(data_path)

    centroids = []
    lichens_name = []
    all_training_vectors = []
    names = []

    for l,lichens in enumerate(lichens_species):
        names.append(lichens)
        print(lichens)
        pth = os.path.join(data_path,lichens)
        lichens_images = [pth + '/'+i for i in os.listdir(pth)]
        lichens_vec = np.zeros((len(lichens_images),64))
        for ii,x in enumerate(lichens_images):
            print(x)
            _,surf_mean = surf.create_surf_descriptors(x,step_size = 25)
            lichens_vec[ii,:] =  surf_mean

        all_training_vectors.append(lichens_vec)

        lichen_centroid = np.zeros(64)

        for i in range(lichens_vec.shape[1]):
            temp = np.zeros(lichens_vec.shape[0])
            for jj in range(lichens_vec.shape[0]):
                temp[jj] = lichens_vec[jj,ii]
            lichen_centroid[ii] = np.mean(temp)

        centroids.append(lichen_centroid)

    np.save('../files/lichens__SURF_centroid',centroids)
    np.save('../files/lichens_SURF_vectors',all_training_vectors)
    np.save('../lichens_name',names)

    return centroids, all_training_vectors








"""
************************************* LEG+COL+CO_OCC **********************
"""


def calculate_color_histogram_and_centroids(data_path = "../final_dataset/"):
    # elencare tutti i tipi diversi di lichene che ho
    lichens_species = os.listdir(data_path)

    centroids = []
    lichens_name = []
    all_training_vectors = []
    names = []

    for l,lichens in enumerate(lichens_species):
        names.append(lichens)
        print(lichens)
        pth = os.path.join(data_path,lichens)
        lichens_images = [pth + '/'+i for i in os.listdir(pth)]
        lichens_vec = []
        for ii,x in enumerate(lichens_images):
            print(x)
            vec_mean = co_occurrence_feature.total_feature(x, 50)
            lichens_vec.append(vec_mean)

        all_training_vectors.append(lichens_vec)
        print()
        lichen_centroid = np.zeros(48)

        for i in range(lichens_vec.shape[1]):
            temp = np.zeros(lichens_vec.shape[0])
            for jj in range(lichens_vec.shape[0]):
                temp[jj] = lichens_vec[jj,ii]
            lichen_centroid[ii] = np.mean(temp)

        centroids.append(lichen_centroid)


    np.save('../files/lichens__COLOR_HISTOGRAM_centroid',centroids)
    np.save('../files/lichens_COLOR_HISTOGRAM_vectors',all_training_vectors)
    np.save('../lichens_color_name',names)

    return centroids, all_training_vectors





BGF_centroid,BGF_training_vectors = calculate_BGF_training_vectors_and_centroids()
SURF_centroids, SURF_training_vectors = calculate_surf_vectors_and_centroids()
HIST_centroid, HIST_training_vectors = calculate_color_histogram_and_centroids()

BGF_centroid = np.load("../files/lichens__BGF_centroid.npy")
BGF_training_vectors = np.load("../files/lichens_BGF_vectors.npy",allow_pickle=True)


BGF_training_vectors[:][0].shape

img = cv2.imread("prova3.jpg")


np.sqrt(256)

bgf_feat = bgf.BGF(img,50,50,50)




res = []
for i in range(0,216*256,216):

    res.append(dst.Eucledian_distance(bgf_feat,cent[i:i+216]))


index = np.argmin(res)

res[index]import cv2
detector = cv2.FeatureDetector_create("Dense")
