
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
import cv2
import math
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage as ndi


from scipy import ndimage
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg as LAsci
from scipy import ndimage as ndi









class filters:



    def __init__(self, image_path, filters_list):
        self.image = io.imread(image_path,as_gray = True)
        if(self.image.shape[0]>2000 or self.image.shape[1]>2000):
            self.image= resize(self.image, (self.image.shape[0]//3, self.image.shape[1]//3),anti_aliasing=True)
        self.image_path = image_path
        self.filters_list = filters_list



    def log_filter(self, sgm, fsize):
        """
        LoG filter
        :param sgm: sigma in Gaussian
        :param fsize: filter size, [h, w]
        :return: LoG filter
        """
        wins_x = int(fsize[1] / 2)
        wins_y = int(fsize[0] / 2)

        out = np.zeros(fsize, dtype=np.float32)

        for x in range(-wins_x, wins_x+1):
            for y in range(-wins_y, wins_y+1):
                out[wins_y+y, wins_x+x] = - 1. / (math.pi * sgm**4.) * (1. - (x*x+y*y)/(2.*sgm*sgm)) * math.exp(-(x*x+y*y)/(2.*sgm*sgm))

        return out-np.mean(out)

    def gabor_filter(self,sgm, theta):
        """
        Gabor filter
        :param sgm: sigma in Gaussian
        :param theta: direction
        :return: gabor filter
        """
        phs=0
        gamma=1
        wins=int(math.floor(sgm*2))
        f=1/(sgm*2.)
        out=np.zeros((2*wins+1, 2*wins+1))

        for x in range(-wins, wins+1):
            for y in range(-wins, wins+1):
                xPrime = x * math.cos(theta) + y * math.sin(theta)
                yPrime = y * math.cos(theta) - x * math.sin(theta)
                out[wins+y, wins+x] = 1/(2*math.pi*sgm*sgm)*math.exp(-.5*((xPrime)**2+(yPrime*gamma)**2)/sgm**2)*math.cos(2*math.pi*f*xPrime+phs)
        return out-np.mean(out)


    def image_filtering(self):
        sub_img = []
        # put the originale image in the list
        sub_img.append(np.float32(self.image))
        for filter in self.filters_list:
            assert (filter[0] == 'log') | (filter[0] == 'gabor'), 'Undefined filter name. '
            if filter[0] == 'log':
                f = self.log_filter(filter[1], filter[2])
                tmp = ndimage.convolve(np.float32(self.image), f, mode='reflect')
                #tmp = np.float32(log_filter(np.float32(img),filter[1]))
                sub_img.append(tmp)
            elif filter[0] == 'gabor':
                f = self.gabor_filter(filter[1], filter[2])
                tmp = ndimage.convolve(np.float32(self.image), f, mode='reflect')
                sub_img.append(tmp)
        return np.float32(np.stack(sub_img, axis=-1))
