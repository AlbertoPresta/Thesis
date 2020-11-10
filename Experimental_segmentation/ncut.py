import numpy as np
import matplotlib.pyplot as plt
from descriptor.binary_gabor_features import BGF
import PIL
from PIL import Image
import cv2
import os
from PIL import Image, ImageOps
from descriptor.co_occurrence_feature import color_hist_vector
import Experimental_segmentation.NcutValue as nct
from scipy.sparse.linalg import eigs
from  scipy.optimize import fmin


img_prova =Image.open("immagini_test/prova3.jpg")
#img_prova = ImageOps.grayscale(img_prova)
img_prova = np.asarray(img_prova)
img_prova = cv2.resize(img_prova,(1000,1000))



def pass_from_coordinate_to_node(i,j,N):
    return (i-1)*N + j

def pass_from_node_to_coordinate(k,N):
    return (k/N,k%N)


def calculate_weights(k,t,T,dd,sigma = 1):
    first_vec = T[k].reshape(-1)
    second_vec = T[t].reshape(-1)
    val = (np.linalg.norm(first_vec-second_vec)/(sigma*sigma))
    distance = calculate_euclidean_distance(k,t,dd,r = 4)
    return np.exp(-val)*distance



def calculate_euclidean_distance(k,t,dd,r = 1.5,sigma_x = 1):
    k_node = pass_from_node_to_coordinate(k,20)
    t_node = pass_from_node_to_coordinate(t,20)

    A = np.array([k_node[0],k_node[1]]).reshape(-1)
    B = np.array([t_node[0],t_node[1]]).reshape(-1)

    distance = np.linalg.norm(A - B)
    dd.append(distance)
    if(distance > r):
        return 0
    else:
        return np.exp(-distance/(sigma_x**2))


def create_feat_mat_and_weight_mat(img_path, DIM_IMG,DIM_SQUARE,dim_feat):
    img_prova =Image.open(img_path)
    img_prova = np.asarray(img_prova)
    img_prova = cv2.resize(img_prova,(DIM_IMG,DIM_IMG))

    x = np.arange(0,DIM_IMG + DIM_SQUARE,DIM_SQUARE)
    y = np.arange(0,DIM_IMG + DIM_SQUARE,DIM_SQUARE)

    num_cells = DIM_IMG//DIM_SQUARE

    T = np.zeros((num_cells,num_cells,dim_feat))
    W = np.zeros((num_cells**2,num_cells**2))


    print('len',len(x))
    for i in range(1,len(x)):
        print(i)
        for j in range(1,len(y)):
            img = img_prova[x[i-1]:x[i],y[i-1]:y[i],:]

            bgf = np.reshape(BGF(img,i,j,50,halflngt = 17, all = True),-1)
            col = np.reshape(color_hist_vector(img,  bins = 16),-1)
            temp = np.reshape(np.concatenate([bgf,col]),-1)
            T[i-1,j-1,:] = temp

    T = T.reshape((num_cells**2,dim_feat))





    dd = []
    for k in range(num_cells**2):
        for t in range(k,num_cells**2):
            res = calculate_weights(k,t,T,dd)
            W[k,t] = res
            W[t,k] = res



    print("% of 0's",(W.reshape(-1).shape[0] - np.count_nonzero(W.reshape(-1)))/W.reshape(-1).shape[0])

    return T,W,dd

T,W,dd = create_feat_mat_and_weight_mat("immagini_test/prova3.jpg", 1000,50,264)







###################################### FIRST CUT ########
seg = np.arange(0,400)                           # the first segment has whole nodes. [1 2 3 ... N]'
id = 'ROOT'                                      # recursively repartition
d = np.sum(W,axis = 1)
D = np.diag(d)
U,S = eigs((D - W), k=2, M=D,  which='SM')
v = S[:,0]
t = np.mean(v)
t = fmin(nct.Ncutvalue,t,args=(v, W , D))
A = np.argwhere(v>t)
B = np.argwhere(v<=t)
ncut = nct.Ncutvalue(t,v,W,D)
ncut
A.shape
W.shape
W[A,A][1]

A[0]
sArea = 20
sNcut = 1.20
A

t = np.array([[1,2,3],[4,5,6],[7,8,9]])

t = t[]


sArea = 20
sNcut = 1.20
rr = [3,5,8,9]
seg[rr][0]
################################################
"""
DEFINISCO LA FUNZIONE VERA E PROPRIA
"""
def NcutPartition(seg, W,sNcut,sArea):

    N = W.shape[0]
    d = np.sum(W,axis = 1)
    D = np.diag(d) # diagonal matrix

    # resolve the generalized system
    U,S = eigs((D - W), k=2, M=D,  which='SM')
    U2 = S[:,0]
    t =np.mean(U2)
    t = fmin(nct.Ncutvalue,t,args = (U2,W,D))
    A = np.argwhere(U2 > t)
    B = np.argwhere(U2 <= t)

    ncut = nct.Ncutvalue(t,U2,W,D)
    print('ncut: ',ncut)
    print('len(A) ',A.shape)
    print('len(B) ',B.shape)
    if len(A)< sArea or len(B) < sArea or ncut > sNcut:
        Seg = seg
        Ncut = ncut

        return Seg, Ncut
    else:
        print("ouuu")
        print(W.shape)

        Wa = extract_weight_matrix(A,W)
        Wb = extract_weight_matrix(B,W)
        print("nuove matrici")
        print(A)
        print("-------")
        print(B)
        segA , ncutA  = NcutPartition(A,Wa,sNcut,sArea)
        segB , ncutB = NcutPartition(B,Wb,sNcut,sArea)
        Seg = [segA, segB]
        Ncut = [ncutA, ncutB]
        return Seg, Ncut


def extract_weight_matrix(a,W):
    print("contruction new matrix")
    print(a.shape)
    W_hat = np.zeros((len(a),len(a)))
    for i in range(len(a)):
        for j in range(i,len(a)):
            W_hat[i,j] = W[a[i],a[j]]
            W_hat[j,i] = W[a[i],a[j]]
    return W_hat

seg


r = np.array(Seg,dtype=object)

len(r[0][0][1]

for i in range(len(Seg)):
    print('----')
    print(Seg[i])
