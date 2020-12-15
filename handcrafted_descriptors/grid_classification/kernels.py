import numpy as np
import os
import cv2




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





def compute_gram_matrix(data,kernel):
    samples,_ = np.shape(data)
    matrix = np.zeros((samples,samples))
    for r in range(samples):
        if(r%400==0):
            print(r)
        for c in range(r,samples):
            tmp = kernel(data[r],data[c])
            matrix[r,c] = tmp
            matrix[c,r] = tmp
    return matrix
