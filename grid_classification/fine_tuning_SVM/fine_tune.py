import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D,Flatten
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
image_size = 100
vgg_s = VGG16(weights='imagenet', include_top=False,input_shape = (image_size,image_size,3))



num_classes = 28
model = models.Sequential()
model.add(vgg_s)
model.add(Conv2D(128,(3,3), activation = 'relu',padding = 'same'))
model.add(MaxPooling2D((2,2),padding = 'same'))
model.add(Conv2D(256,(3,3),activation = 'relu',padding = 'same'))
model.add(MaxPooling2D((2,2),padding = 'same'))
#model.add(GlobalAveragePooling2D())
#model.add(Dense(65,activation = 'relu'))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(num_classes,activation = 'softmax'))
model.layers[0].trainable = False
optimizer = Adam(lr = 0.00006)
model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],optimizer=optimizer)



model.summary()

labels = os.listdir("grid_classification/patches/train")

train_dir = "grid_classification/patches/train/"
from grid_classification.fine_tuning_SVM import utils



list_of_images = utils.list_of_path(labels,train_dir)
train_data,train_labels = utils.read_and_process_images(list_of_images,labels,dimension=100)

from keras.utils import to_categorical
from keras.preprocessing import image
train_images =  []
for im in list_of_images:
    temp_img=image.load_img(im,target_size=(100,100))
    temp_img=image.img_to_array(temp_img)
    train_images.append(temp_img)



train_images=np.array(train_images)
train_img=preprocess_input(train_images)


train_labels_dummy=to_categorical(train_labels,28)


from keras.callbacks import EarlyStopping
from keras.callbacks import History

history = History()
earlyStopping = EarlyStopping(min_delta=0.00,patience = 3)
model.fit(train_img,train_labels_dummy,batch_size=64,epochs=10,validation_split=0.15,shuffle=True,callbacks=[earlyStopping,history])

train_labels_dummy.shape



############################ EVALUATION ############################################

test_dir = "grid_classification/patches/test/"
from grid_classification.fine_tuning_SVM import utils



list_of_images = utils.list_of_path(labels,test_dir)
test_data,test_labels = utils.read_and_process_images(list_of_images,labels,dimension=100)

test_images =  []

for im in list_of_images:
    temp_img=image.load_img(im,target_size=(100,100))
    temp_img=image.img_to_array(temp_img)
    test_images.append(temp_img)


test_images=np.array(test_images)
test_img=preprocess_input(test_images)

test_labelsdummy=to_categorical(test_labels,28)


ev = model.evaluate(test_img,test_labelsdummy)
########## save model


model.save('grid_classification/fine_tuning_SVM/my_model')


%%time
y_pred = model.predict_classes(test_img[:1,:,:,:])


y_pred[0]
