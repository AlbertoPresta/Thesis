import numpy as np
from skimage.feature.texture import greycomatrix, greycoprops
import cv2
import os
import matplotlib.pyplot as plt





def Legendre_pol(p,x):
    # base case
    if(p==1):
        return x
    elif p == 0:
        return 1
    else:
        res = ((2*p -1)/p)*x*Legendre_pol(p-1,x) - ((p-1)/p)*Legendre_pol(p-2,x)
        return res


def Legendre_moments(p,q,img):
    M = img.shape[0]
    N = img.shape[1]
    Lam_pq = ((2*p + 1)*(2*q + 1))/(M*N)
    L_pq = 0
    for i in range(M):
        for j in range(N):
            x_hat = (2*i - M + 1)/(M-1)
            y_hat = (2*j - N + 1)/(N-1)
            L_pq += Legendre_pol(p,x_hat)*Legendre_pol(q,y_hat)*img[i,j]
    return Lam_pq*L_pq



def leg_tuple(level):
    res = []
    for i in range(level + 1):
        for j in range(level + 1):
            if i +j <=level:
                res.append((i,j))
    return res


def Legendre_vector(img, level):
    res = []
    tup = leg_tuple(level)
    for i in range(3):
        im = img[:,:,i]
        for t in tup:
            res.append(Legendre_moments(t[0],t[1],im))
    return np.array(res)



def co_occurrence_vector(img, distance = [1], direction = [0, np.pi/2, 3*np.pi/2]):
    res = []
    for i in range(3):
        im = img[:,:,i]
        co_matrix = greycomatrix(im, distance, direction, levels=256)
        contrast = greycoprops(co_matrix, 'contrast')
        dissimilarity = greycoprops(co_matrix, 'dissimilarity')
        homogeneity = greycoprops(co_matrix,  'homogeneity')
        energy = greycoprops(co_matrix, 'energy')
        correlation = greycoprops(co_matrix, 'correlation')
        ASM = greycoprops(co_matrix, 'ASM')
        total = np.concatenate([contrast,dissimilarity,homogeneity,energy,correlation,ASM],axis = 1)
        res.append(total)
    return np.array(res).ravel()




def color_hist_vector(img, bins = 16):
    res = []
    for i in range(3):
        im =img[:,:,i]
        M = im.shape[0]
        N = im.shape[1]
        c = plt.hist(img.ravel(), bins=16)
        plt.close()
        norm = np.linalg.norm(c[0])
        hist_norm = c[0]/norm
        res.append(hist_norm)
    return np.array(res).ravel()


def crop_image(img_path,new_width = 400, new_height = 400):
    img = cv2.imread(img_path)
    print("original_shape: ",img.shape)
    img = cv2.resize(img,(1000,1000))
    #plt.imshow(img)
    #print(img.shape)
    height, width,_ = img.shape
    height = height//2
    width = width//2

    left = (width - new_width)
    top = (height - new_height)
    right = (width + new_width)
    bottom = (height + new_height)


    img = img[top:bottom,left:right,:]
    print(img.shape)
    return img






def total_feature(img_path, size,level = 3, bins = 16, distance = [1], direction = [0, np.pi/2, 3*np.pi/2]):

    img = crop_image(img_path)

    x_sizes = np.arange(0,img.shape[0],size)
    y_sizes = np.arange(0,img.shape[1],size)
    res = []
    for i in x_sizes:
        for j in y_sizes:
            im = img[i:i + size,j:j+size,:]
            #legendre = Legendre_vector(im,level)
            histograms = color_hist_vector(im,  bins = bins)
            #co_occurrence = co_occurrence_vector(im,distance = distance, direction = direction)
            res.append(histograms)
            #res.append(np.concatenate([histograms,co_occurrence]))
    # calcolo la media
    res = np.array(res)
    media =  np.zeros(48)
    for jj in range(res.shape[1]):
        tmp = np.zeros(256)
        for ii in range(res.shape[0]):
            tmp[ii] = res[ii,jj]
        media[jj] = np.mean(tmp)

    return res, media




""""""""
