import numpy as np
from scipy.ndimage.filters import convolve
import cv2
from matplotlib import pyplot as plt
from descriptor.sift import DoG as dog
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt


def rotate_window(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    t =  np.squeeze((R @ (p.T-o.T) + o.T).T)
    return math.ceil(t[0]), math.ceil(t[1])
















def find_main_direction(hist, num_bins,range):
    values = range + 5
    maximum_index = np.argmax(hist)
    x_0 = values[maximum_index - 1]
    x_1 = values[maximum_index + 1]
    x = np.linspace(x_0,x_1,num = 3,endpoint = True)
    y = None
    if(maximum_index == 0):
        media = (hist[maximum_index] + hist[maximum_index + 1])/2
        y = np.array([hist[maximum_index],media ,hist[maximum_index + 1]])
    elif(maximum_index== len(hist) -1):
        media = (hist[maximum_index -1] + hist[maximum_index])/2
        y = np.array([hist[maximum_index -1],media ,hist[maximum_index]])
    else:
        y = np.array([hist[maximum_index - 1],hist[maximum_index],hist[maximum_index + 1]])
    f2 = interp1d(x, y, kind='quadratic')
    xnew = np.linspace(x_0,x_1, num=1001, endpoint=True)
    c = f2(xnew)
    return xnew[np.argmax(c)]

def assign_orientation(img, kps, num_bins, size,sigma):
    kernel = dog.gaussian_filter(sigma)
    img = convolve(img,kernel)
    # consider only sub image
    cx = kps[0]
    cy = kps[1]
    bin_width = 360//num_bins
    sz = size//2
    img_patch = img[cx - sz:cx +sz,cy - sz:cy + sz]

    gx = cv2.Sobel(img_patch,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(img_patch,cv2.CV_64F,0,1,ksize=3)

    magnitude =  cv2.magnitude(gx, gy)
    magnitude = gaussian_filter(magnitude,sigma*1.5)

    directions = cv2.phase(gx, gy,angleInDegrees=True)

    magnitude = magnitude.reshape(-1)
    directions = directions.reshape(-1)

    hist,range = np.histogram(directions, bins=num_bins,range=(0,360), weights=magnitude)

    main_dir = find_main_direction(hist,num_bins,range)



    return  main_dir






"""
img = cv2.imread("../prova2.jpg")
gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(400)
step_size = 20
kp = [cv2.KeyPoint(x, y, step_size) for y in range(10, gray.shape[0], step_size)
                  for x in range(10, gray.shape[1], step_size)]


sig = dog.generate_sigmas()
dg, gaussian_images = dog.dogs(gray,sig)
sigma, index = dog.maximum(dg,sig,(1200,1000))

# assign orientation of each keypoint
for k in kp:
    x = int(k.pt[1])
    y = int(k.pt[0])
    dr = assign_orientation(gray, (x,y), 36, 16,1)
    print(y," ",x)
    print(dr)
    k.angle = dr

%%time
surfs = []
for k in kp:
    x = int(k.pt[1])
    y = int(k.pt[0])
    print(y," ",x)
    sigma, index = dog.maximum(dg,sig,(x,y))
    dr = assign_orientation(gray, (x,y), 36, 16,1)
    print(dr)
    k.angle = dr
    _,c = surf.compute(gaussian_images[index],[k])
    surfs.append(c)

"""



#kpp = sift.detect(gray,None)

#a,b,c) = unpackSIFTOctave(kpp[3000])


#img=cv2.drawKeypoints(gray,kp, img)
#plt.figure(figsize=(20,10))
#plt.imshow(img)
#plt.show()


#dense_feat,c = sift.compute(gray, kp)
