import numpy as np
import os
from descriptor import binary_gabor_features as bgf
from descriptor import distance as dst
from descriptor import co_occurrence_feature
from descriptor import surf
import cv2
import matplotlib.pyplot as plt



bgf_vectors = np.zeros(216)
uno = np.ones(216)
v = np.concatenate([uno,bgf_vectors])


v

bgf_vectors = np.load('../files/lichens_BGF_vectors.npy', allow_pickle = True)
bgf_centroid = np.load('../files/lichens__BGF_centroid.npy', allow_pickle = True)



dst.Eucledian_distance(prima_classe[1,:],prima_classe[2,:])

surf_vectors = np.load('../files/lichens_SURF_vectors.npy', allow_pickle = True)
surf_centroid = np.load('../files/lichens__SURF_centroid.npy',allow_pickle = True)

surf_vectors[14].shape
# calcolo la within_cluster dissimilarity

#prendiamo la lcasse 0

prima_classe = bgf_vectors[20]
prima_classe_surf = surf_vectors[1]

cont = 0
c = 0
for i in range(prima_classe.shape[0]):
    x = prima_classe[i,:]
    for j in range(i+1,prima_classe.shape[0]):
        c = c +1
        y = prima_classe[j,:]
        dist = dst.Eucledian_distance(x,y)
        print(dist)
        cont = cont + dist

cont/c



def compute_within_dissimilarity()
