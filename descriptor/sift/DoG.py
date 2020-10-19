import numpy as np
from scipy.ndimage.filters import convolve
import cv2
from matplotlib import pyplot as plt


def gaussian_filter(sigma):
    """
    function which creates gaussian filer of dimension
    size, which is calculated below
    """
    size = 2*np.ceil(3*sigma + 1)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
    return g/g.sum()


# PER ORA CONSIDERO SOLO UN OCTAVE CON 8 LIVELLI (PARTO DA SIGMA = 1.6)
# K = RADICE OTTAVA DI DUE



def generate_octave(init_level, s, sigma):
    octave = [init_level]
    k = 2**(1/s)
    kernel = gaussian_filter(k*sigma)
    ksig = k*sigma
    for i in range(s+2):
        next_level = convolve(octave[-1],kernel)
        octave.append(next_level)
        ksig *=k
    return octave





def generate_sigmas( l = 7,levels = 2):
    val = np.arange(l, 0,-1)
    steps = [1]
    for i in range(1,levels + 1):
        steps.append(2*i)
    sigmas = [1]
    for ii in steps:
        for jj in val:
            sigmas.append(ii*pow(2,1/jj))
    # voglio anche per 16
    sigmas.append(2*steps[-1]*pow(2,1/val[0]))
    return sigmas


#sig
def dogs(img, sigmas):
    res = []
    gf = gaussian_filter(sigmas[0])

    gaussian_images = [convolve(img,gf)]
    for i in range(1,len(sigmas)):
        kernel_2 = gaussian_filter(sigmas[i])
        image_2 = convolve(img,kernel_2)
        gaussian_images.append(image_2)

        kernel_1 = gaussian_filter(sigmas[i-1])
        image_1 = convolve(img,kernel_1)

        res.append(image_2 - image_1)
    return res, gaussian_images



#t = dogs(img,sig)

def maximum(t,sigmas,kp):
    x = kp[0]
    y = kp[1]
    values = []
    for i in range(len(t)):
        image = t[i]
        values.append(image[x,y])
    return sigmas[np.argmax(values)], np.argmax(values)


#c,d = maximum(t,sig,(600,100))













######################################################################

def gen_octave(img,s, sigma):
    k = 2**(1/s)
    sigmas = [k*sigma]
    tmp = k*sigma
    for i in range(s - 1):
        sigmas.append(tmp*k)
        tmp = tmp*k

    octave = []
    for i in range(len(sigmas)):
        kernel = gaussian_filter(sigmas[i])
        image = convolve(img,kernel)
        octave.append(image)
    return octave


def gen_pyr(img,levels,s,sigma):
    pyr = []
    for i in range(levels):
        pyr.append(gen_octave(img,s,sigma))
        img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
    return pyr



def generate_DoG_pyramid(img, level, s, sigma):
    # generate gaussian pyramid
    gaussian_pyr = gen_pyr(img, level, s, sigma)
    dog_pyr = []
    for i in range(level):
        temp_pyr = []
        for j in range(1,s):
            temp_pyr.append(gaussian_pyr[i][j-1] - gaussian_pyr[i][j])
        temp_pyr = np.asarray(temp_pyr)
        dog_pyr.append(temp_pyr)
    return dog_pyr


#img = cv2.imread("../prova2.jpg",0)


#oc = generate_DoG_pyramid(img,4,7,1.6)


def find_maximum_across_scales_of_a_point(oc,kp):
    x = kp[0]
    y = kp[1]
    values = np.zeros([4,6])
    v = [1,2,4,8]
    for  ii in range(len(oc)):
        #temp_values = []
        for jj in range(len(oc[ii])):
            image = oc[ii][jj]
            #temp_values.append(image[x//v[ii],y//v[ii]])
            values[ii,jj] = image[x//v[ii],y//v[ii]]
        #values.append(temp_values)
    return values




#f = find_maximum_across_scales_of_a_point(oc,(600,100))
