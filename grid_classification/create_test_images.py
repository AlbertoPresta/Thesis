import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from scipy.io import savemat,loadmat
import pickle

"""
IMMAGINI DIFFICILI, CI DVE ESSERE UNA CLASSIFICAZIONE PIXL PER PIXEL DELL'IMMAGINE
"""

def save_images_in_npy_file(pth):
    lista_licheni = os.listdir(pth)
    res = np.zeros([40,1000,1000,3])
    lichene_type = []
    cont = 0
    for i,lich in enumerate(lista_licheni):
        lista_immagini_pth = os.path.join(pth,lich)
        lista_immagini = os.listdir(lista_immagini_pth)
        for j,img in enumerate(lista_immagini):
            img_pth = os.path.join(lista_immagini_pth,img)
            immagine = cv2.imread(img_pth)
            immagine = cv2.cvtColor(immagine, cv2.COLOR_RGB2BGR)
            x_dim = immagine.shape[0]//10
            y_dim = immagine.shape[1]//10
            immagine = immagine[x_dim:immagine.shape[0] - x_dim,y_dim:immagine.shape[1] - y_dim,:]
            immagine = cv2.resize(immagine,(1000,1000))



            res[cont,:,:,:] = immagine
            cont = cont +1
            print(cont)
            lichene_type.append(lich)
    res_fin = []
    for t in res:
        res_fin.append(t.astype(np.uint8))
    res_fin = np.array(res_fin)
    return res_fin, lichene_type




def generate_collages(textures,batch_size=1,segmentation_regions=10,anchor_points=None):
    # Returns a batch of mixed texture, reference mask, and reference texture index
    N_textures = textures.shape[0]
    img_size= textures.shape[1]
    masks, n_points = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    print('*********')
    print(textures[textures_idx[0]].shape)
    print((textures[textures_idx[0]] * masks[:,:,:,0:0+1]).shape)
    batch = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions))
    print('end batching!')
    ref_idx = [np.random.randint(i) for i in n_points]
    batch_res = []
    for b in batch:
      bt = b.astype(np.uint8)
      batch_res.append(bt)
    return np.array(batch_res), masks[range(batch_size),:,:,ref_idx].reshape((batch_size, 1000, 1000, 1)), textures_idx[ref_idx,range(batch_size)]

def generate_random_masks(img_size=1000, batch_size=1, segmentation_regions=10, points=None):
    xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size))

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
        # n_points = [segmentation_regions] * batch_size
        points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    print(points[0].shape)
    masks = []
    for b in range(batch_size):
        print('------> ',b)
        dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]]
        voronoi = np.argmin(dists_b, axis=0)
        masks_b = np.zeros((img_size, img_size, segmentation_regions))
        for m in range(segmentation_regions):
            masks_b[:,:,m][voronoi == m] = 1
        masks.append(masks_b)
    return np.stack(masks), n_points

def generate_validation_collages(N=100):
    textures = np.load('val_texture.npy')
    collages = generate_collages(textures, batch_size=N)
    print('END CLLAGING')
    np.save('validation_collages.npy', collages[0])
    return collages[0],collages[1]





"""
IMMAGINI SQUADRATE
"""

def create_synthetic_image(res,lichene_type, number_cols = 1,max_number_row = 2):
    cols = random.sample(range(100, 900,100), number_cols)
    cols.append(0)
    cols.append(1000)
    cols.sort()
    image = np.zeros([1000,1000,3])
    species = []
    for i in range(1,len(cols)):
        nmb_rows = random.sample(range(1,max_number_row),1)
        rows = random.sample(range(100, 900,100), nmb_rows[0])
        rows.append(0)
        rows.append(1000)
        rows.sort()
        # scelgo le immagini casuali
        immagini = []
        num = random.sample(range(0,len(res)), nmb_rows[0] + 1)
        for j in num:
            immagini.append(res[j])
            species.append(lichene_type[j])
        for k in range(1,len(rows)):
            image[cols[i-1]:cols[i],rows[k-1]:rows[k],:] = immagini[k-1][cols[i-1]:cols[i],rows[k-1]:rows[k],:]

    return image.astype(np.uint8),species

index


"""
IMMAGINI PIU'SEMPLICI: PER ORA PRENDO IN CONSIDERAZIONE QUESTE! IMMAGINE DIVISA IN 4 CON LICHENI DIVERSI
VEDO COME SI RIESCE A CLASSIFICARE IL TUTTO.
"""
index = c.index('Arthonia_radiata')
species = ['Arthonia_radiata','Caloplaca_cerina','Candelariella_reflexa','Candelariella_xanthostigma','Chrysothrix_candelaris','Flavoparmelia_caperata','Gyalolechia_flavorubescens','Hyperphyscia_adglutinata'
        ,'Lecanora_argentata','Lecanora_chlarotera','Lecidella_elaeochroma','Melanelixia_glabratula'
        ,'Phaeophyscia_orbicularis','Physcia_biziana','Physconia_grisea','Ramalina_farinacea','Ramalina_fastigiata','Xanthomendoza_fallax','Xanthomendoza_fulva','flavoparmenia_soredians']
c = list(np.array(species)


classes = np.arange(0,20,1)

classes

def create_4by4_images(res, nome_immagine,lichene_type):
    tipi_lichene = list(np.unique(lichene_type))
    img_numb = random.sample(range(res.shape[0]),4)
    image = np.zeros([1000,1000,3])
    true_classification =np.zeros((10,10))
    cord_cl = [[0,5,0,5],[0,5,5,10],[5,10,0,5],[5,10,5,10]]
    cord = [[0,500,0,500],[0, 500, 500, 1000],[500,1000,0 ,500],[500,1000,500,1000]]
    for i,numb in enumerate(img_numb):
        print('---->',numb)
        crd = cord[i]
        crd_cl = cord_cl[i]
        im = res[numb,:,:,:]
        lich = lichene_type[numb]
        print(lich)
        image[crd[0]:crd[1],crd[2]:crd[3],:] = im[250:750,250:750,:]
        true_classification[crd_cl[0]:crd_cl[1],crd_cl[2]:crd_cl[3]] = tipi_lichene.index(lich)
    np.save("grid_classification/descriptors/labels/" + nome_immagine +".npy", true_classification)
    cv2.imwrite("../images_matlab_test/images/" + nome_immagine +".jpg",cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))





"""
MAIN
"""
lichene_type[38]


res, lichene_type = save_images_in_npy_file('../final_dataset/test/')
for ii in range(20):
    print(ii)
    nome_immagine = 'text_image_N' + str(ii+1)
    create_4by4_images(res, nome_immagine, lichene_type)


c = np.load('grid_classification/descriptors/labels/text_image_N2.npy')




os.listdir("grid_classification/descriptors/labels/")
