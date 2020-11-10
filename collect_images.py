import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random






def save_images_in_npy_file(pth):
    lista_licheni = os.listdir(pth)
    res = np.zeros([58,1000,1000,3])
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




res, lichene_type = save_images_in_npy_file('../final_dataset/test/')



plt.imshow(res[0])
random.sample(res, 2)


np.save('val_texture.npy',res)

plt.imshow(clg[0])

clg,mask = generate_validation_collages()

for i,c in enumerate(clg):
    nome = 'text_image_' + str(i) + '.jpg'
    nome_completo = os.path.join('../synthetic_images',nome)
    tmp_c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    cv2.imwrite(nome_completo,tmp_c)



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





im,spec = create_synthetic_image(res,lichene_type)





cv2.imwrite('img_prova.jpg',cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))

species_for_images = {}

for i in range(100):
    nome = '_squared_text_image_' + str(i) + '.jpg'
    nome_completo = os.path.join('../synthetic_images_2',nome)
    img,spec = create_synthetic_image(res,lichene_type)
    species_for_images[nome_completo] = spec
    tmp_c = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(nome_completo,tmp_c)



np.save('legend_squared_images.npy',species_for_images)
plt.figure(figsize=(20,10))
plt.imshow(img)





image[0:500,0:500,:].shape

res[num[0]].shape



image = np.zeros([1000,1000,3])

num = random.sample(range(0,len(res)),4)
image[0:500,0:500,:] = res[num[0]][250:750,250:750,:]
image[500:1000,0:500,:] = res[num[1]][250:750,250:750,:]
image[0:500,500:1000,:] = res[num[2]][250:750,250:750,:]
image[500:1000,500:1000,:] = res[num[3]][250:750,250:750,:]
plt.imshow(image.astype(np.uint8))

img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imwrite('tst.jpg',image)
