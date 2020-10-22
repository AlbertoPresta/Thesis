"""
In this script i will analize lichens class, by calculating within class distance and
between class distance.
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from numpy import asarray
from PIL import Image





# calculate  descriptors for each class
def calculate_dataset(pth, CENTERS = [(300,300),(300,700),(700,300),(700,700)],SIZE = 200):
    sift = cv2.xfeatures2d.SIFT_create()
    lichens = os.listdir(pth)
    class_features = {}
    dic_label = {} # salvo corrispondenza classe numero qua
    l = 0
    cropped = {}
    for lichen in lichens:
        print("--------- ",lichen," -----------")
        print(l)
        dic_label[lichen] = l
        lichen_path = os.path.join(pth,lichen)
        lichen_sing_images = os.listdir(lichen_path)
        descript = []
        cropped_images = []
        for img in lichen_sing_images:

            if(img == ".DS_Store"):
                continue
            img_path = os.path.join(lichen_path,img)
            print(img_path)
            im = Image.open(img_path)
            im = asarray(im)
            im = cv2.resize(im,(1000,1000))
            #im = im / 255.0
            for ii,crd in enumerate(CENTERS):
                crop_img = im[crd[0]-SIZE:crd[0] +SIZE,crd[1] - SIZE:crd[1] +SIZE,:]
                #crop_img = crop_img/255
                #crop_img = asarray(crop_img)
                print(crop_img.shape)
                cropped_images.append(crop_img)
                kp = cv2.KeyPoint(crd[0],crd[1],SIZE)
                _,feaArrSingle_R = sift.compute(crop_img[:,:,0],[kp])
                _,feaArrSingle_G = sift.compute(crop_img[:,:,1],[kp])
                _,feaArrSingle_B = sift.compute(crop_img[:,:,2],[kp])

                feaArrSingle_R =  feaArrSingle_R.reshape(-1)
                feaArrSingle_G =  feaArrSingle_G.reshape(-1)
                feaArrSingle_B =  feaArrSingle_B.reshape(-1)
                #rint(feaArrSingle_R.shape)
                temp = np.concatenate([feaArrSingle_R,feaArrSingle_G,feaArrSingle_B])
                descript.append(temp)
        class_features[lichen] = descript
        l = l + 1
        cropped[lichen] = cropped_images

    return class_features, dic_label,cropped

dic_label
pth = "../final_dataset/train"
class_features, dic_label , cropped= calculate_dataset(pth)

def visualize_image_patches(lichen_name, crop_img = cropped):
    images = crop_img[lichen_name]


    fig=plt.figure(figsize=(100, 100))
    columns = len(images)//4
    rows = len(images) - columns
    for i in range(len(images)):
        img = images[i]
        fig.add_subplot(rows, columns, i +1)
        plt.imshow(img)
    plt.show()

visualize_image_patches('Lecanora_argentata')

dic_label

def calculate_within_complete_and_single_distance(lichen, class_features = class_features):
    lichen_list = np.array(class_features[lichen])
    within_dist = np.zeros((lichen_list.shape[0],lichen_list.shape[0]))
    lista = []
    for i in range(lichen_list.shape[0]):
        for j in range(i+1,lichen_list.shape[0]):
            tmp = np.linalg.norm(lichen_list[i,:] - lichen_list[j,:])
            lista.append(tmp)
            within_dist[i,j] = tmp
            within_dist[j,i] = tmp
    return within_dist, np.max(lista), np.min(lista) , np.mean(lista)



wl , compl, sing = calculate_within_complete_and_single_distance('Lecanora_argentata', class_features = class_features)

compl


# calculate complete distance between two classes
def calculate_complete_and_single_distance(lichen_1,lichen_2,class_features = class_features):
    lichen1_list = np.array(class_features[lichen_1])
    lichen2_list = np.array(class_features[lichen_2])
    distances = np.zeros((lichen1_list.shape[0],lichen2_list.shape[0]))
    for i in range(lichen1_list.shape[0]):
        for j in range(lichen2_list.shape[0]):
            tmp = np.linalg.norm(lichen1_list[i,:] - lichen2_list[j,:])
            distances[i,j] = tmp
            #distances[j,i] = tmp

    return distances , np.max(distances), np.min(distances), np.mean(distances)

distances,complete_distance, single_distance = calculate_complete_and_single_distance('Candelariella_xanthostigma','Lecanora_argentata')

single_distance


def plot_distances(lichen_name,class_features = class_features,type = 'single'):
    if(type=='complete'):
        lichens = []
        within_single = None
        distances_vector = []
        for lich in list(class_features.keys()):
            if(lich==lichen_name):
                _,tmp,_ ,_= calculate_within_complete_and_single_distance(lichen_name, class_features = class_features)
                within_single = tmp
            else:
                lichens.append(lich)
                _,tmp, _, _ = calculate_complete_and_single_distance(lichen_name,lich)
                distances_vector.append(tmp)

        line = []
        for i in range(len(lichens)):
            line.append(within_single)
        fig=plt.figure(figsize=(10, 5))
        plt.plot(lichens,line,label = 'within class distance' )
        plt.plot(lichens,distances_vector,'r+')
        plt.plot(lichens,distances_vector,label='single class distance')
        plt.tick_params(axis='x', rotation=90)
        plt.xlabel("lichens species")
        plt.ylabel("single class distance (Eucledian)")
        plt.title(lichen_name + " single Distance between other lichens")
        plt.legend(loc ='upper right')

        plt.grid()
        plt.savefig("grid_classification/plot_classes_distance/complete_distance/" + lichen_name + "_complete_distance.jpg")
        plt.show()
    elif(type=='single'):
                lichens = []
                within_single = None
                distances_vector = []
                for lich in list(class_features.keys()):
                    if(lich==lichen_name):
                        _,_,tmp,_ = calculate_within_complete_and_single_distance(lichen_name, class_features = class_features)
                        within_single = tmp
                    else:
                        lichens.append(lich)
                        _,_, tmp,_ = calculate_complete_and_single_distance(lichen_name,lich)
                        distances_vector.append(tmp)

                line = []
                for i in range(len(lichens)):
                    line.append(within_single)
                fig=plt.figure(figsize=(10, 5))
                plt.plot(lichens,line,label = 'within class distance' )
                plt.plot(lichens,distances_vector,'r+')
                plt.plot(lichens,distances_vector,label='single class distance')
                plt.tick_params(axis='x', rotation=90)
                plt.xlabel("lichens species")
                plt.ylabel("single class distance (Eucledian)")
                plt.title(lichen_name + " single Distance between other lichens")
                plt.legend(loc ='upper right')

                plt.grid()
                plt.savefig("grid_classification/plot_classes_distance/single_distance/" + lichen_name + "_single_distance.jpg")
                plt.show()
    elif(type=='mean'):
        lichens = []
        within_single = None
        distances_vector = []
        for lich in list(class_features.keys()):
            if(lich==lichen_name):
                _,_,_,tmp = calculate_within_complete_and_single_distance(lichen_name, class_features = class_features)
                within_single = tmp
            else:
                lichens.append(lich)
                _,_,_,tmp = calculate_complete_and_single_distance(lichen_name,lich)
                distances_vector.append(tmp)

        line = []
        for i in range(len(lichens)):
            line.append(within_single)
        fig=plt.figure(figsize=(10, 5))
        plt.plot(lichens,line,label = 'within class distance' )
        plt.plot(lichens,distances_vector,'r+')
        plt.plot(lichens,distances_vector,label='single class distance')
        plt.tick_params(axis='x', rotation=90)
        plt.xlabel("lichens species")
        plt.ylabel("single class distance (Eucledian)")
        plt.title(lichen_name + " single Distance between other lichens")
        plt.legend(loc ='upper right')

        plt.grid()
        plt.savefig("grid_classification/plot_classes_distance/average_distance/" + lichen_name + "_mean_distance.jpg")
        plt.show()






lichen = list(dic_label.keys())

for l in lichen:
    print(l)
    plot_distances(l,class_features = class_features,type = 'single')
    plot_distances(l,class_features = class_features,type = 'complete')
    plot_distances(l,class_features = class_features,type = 'mean')
