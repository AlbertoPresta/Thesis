import numpy as np
import os
import cv2
import descriptor.distance as dst
import descriptor.binary_gabor_features as bgf
import matplotlib.pyplot as plt
from descriptor.co_occurrence_feature import total_feature
import operator
import descriptor.sift.DoG as dog
import descriptor.surf as surf

%matplotlib inline







def create_histogram(img_path,center_x,center_y,size ,hlf,sig,dg,gaussian_images,ft = "../files/lichens__BGF_centroid.npy", nm = "../files/lichens_name.npy"):

    feature =  np.load(ft, allow_pickle = True)
    print(feature.shape)
    name = np.load(nm, allow_pickle = True)
    img = cv2.imread(img_path)

    left_x = max(0,center_y - size)
    right_x = min(center_y + size,img.shape[0])
    left_y = max(0,center_x - size)
    right_y = min(center_x + size,img.shape[1])
    if(ft == "features/color_text_shape.npy"):
        img = img[left_x:right_x, left_y: right_y]
        col_text_feat = total_feature(img)
        print(col_text_feat.shape)
        p_list = []

        for i in range(col_text_feat.shape[0]):
            p_list.append(col_text_feat[i])

        dist = []
        for i,ft in enumerate(feature):
            dis = dst.Eucledian_distance(ft,p_list)
            dist.append(dis)

        summa = 0
        for d in dist:
            summa = summa + d

        for i in range(len(dist)):
            dist[i] = dist[i]/summa

        return dist, name

    elif(ft == "features/LBP_feature_hist.npy"):
        col_text_feat = LBP.LBP_features(img_path,True,sub = True, coord = [left_x, right_x, left_y,right_y])[0]
        bins = np.arange(0,243)
        weightsa = np.ones(len(col_text_feat)) / len(col_text_feat)
        p_list,_ = np.histogram(col_text_feat,bins,weights= weightsa)

        dist = []
        for i,ft in enumerate(feature):

            dis = dst.Eucledian_distance(ft,p_list)
            dist.append(dis)

        summa = 0
        for d in dist:
            summa = summa + d

        for i in range(len(dist)):
            dist[i] = dist[i]/summa


        return dist,name
    elif(ft == "../files/lichens__BGF_centroid.npy"):
        print("GABOR")
        gabor = bgf.BGF(img, center_x, center_y, size, halflngt = hlf,all = False)
        img = img[left_x:right_x, left_y: right_y]
        #col_text_feat = total_feature(img)
        #tot = np.concatenate([gabor[:,0],col_text_feat])

        p_list = []
        for i in range(gabor.shape[0]):
            p_list.append(gabor[i])

        dist = []
        for i,ft in enumerate(feature):
            dis = dst.Eucledian_distance(ft,p_list)
            dist.append(dis)

        summa = 0
        for d in dist:
            summa = summa + d

        for i in range(len(dist)):
            dist[i] = dist[i]/summa

        return dist,name
    elif(ft == "../files/lichens__SURF_centroid.npy"):

        srf = surf.calculate_surf_in_a_specific_point(img_path,(center_x,center_y),dg,sig,gaussian_images,thresh = 400)

        p_list = []
        for i in range(srf.shape[1]):
            p_list.append(srf[0][i])

        dist = []
        for i,ft in enumerate(feature):
            dis = dst.Eucledian_distance(ft,p_list)
            dist.append(dis)

        summa = 0
        for d in dist:
            summa = summa + d

        for i in range(len(dist)):
            dist[i] = dist[i]/summa

        return dist,name




    else:
        return [],[]

"""image = cv2.imread("prova1.jpg")
image = cv2.resize(image, (512,512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sig = dog.generate_sigmas()
dg, gaussian_images = dog.dogs(gray,sig)
surf = surf.calculate_surf_in_a_specific_point("prova1.jpg",(100,100),dg,sig,gaussian_images,thresh = 400)

surf[0][1]

"""

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    img_path = param[0]
    size = param[1]
    hlf = param[2]
    features = param[3]
    sig = param[4]
    dg = param[5]
    gaussian_images = param[5]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN :
        print("coordinate: (",x,",",y,")")
        dist,name = create_histogram(img_path, x, y, size, hlf, sig,dg,gaussian_images,ft = features)
        dist = np.asarray(dist)

        print(dist.shape)
        plt.figure(figsize=(10, 5))

        plt.bar(np.arange(0,29), dist, align='edge', width=0.5,log= True)
        plt.xticks(np.arange(len(name)), name,rotation='vertical')
        plt.show();

        print("winning_class")
        d = dict(zip(name, dist))
        sorted_d = sorted(d.items(), key=operator.itemgetter(1))
        for i in range(10):
            print(sorted_d[i])







image = cv2.imread("prova3.jpg")
image = cv2.resize(image, (512,512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sig = dog.generate_sigmas()
dg, gaussian_images = dog.dogs(gray,sig)

# displaying the image
cv2.imshow('image', image)
param = ["prova2.jpg",50,7,"../files/lichens__SURF_centroid.npy",sig,dg,gaussian_images]
# setting mouse hadler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event,param)
kv = cv2.waitKey(0)
if(kv==27):
    print("destroy image")
    cv2.destroyAllWindows()


cv2.destroyAllWindows()
