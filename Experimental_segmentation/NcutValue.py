import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2


def obtain_vector(c,thres):
    tmp = (c>thres).reshape(-1)
    res = np.zeros(tmp.shape[0])
    for i in range(res.shape[0]):
        if(tmp[i]==False):
            res[i] = 0
        else:
            res[i]=1
    return res.reshape(-1)

def do_sum(d,x):
    res = 0
    if(d.shape[0]!=x.shape[0]):
        print("lunghezze diverse!")
        return res
    else:
        for i in range(x.shape[0]):
            if(x[i] > 0):
                res = res + d[i]
            else:
                continue
    return res



def Ncutvalue(t, U2, W , D):
    x = obtain_vector(U2,t)
    x = (2 * x ) -1
    d = np.diag(D)

    k = do_sum(d,x)/np.sum(d)
    b = k / (1 -k)
    y = (1 +x) -b*(1-x)
    A = np.dot(np.dot(y.T,D-W),y)
    B = np.dot(np.dot(y.T,D),y)
    return A/B
