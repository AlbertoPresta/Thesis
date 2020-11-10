import numpy as np
import cv2
import os


os.listdir()

lichens = 'grid_classification/patches/test'


for lich in os.listdir(lichens):
    path = lichens + '/' + lich
    img = os.listdir(path)
    for im in img:
        f_im = im.split('.')[0]
        os.rename(path + '/' + im, path + '/' + f_im + 'test.jpg')
