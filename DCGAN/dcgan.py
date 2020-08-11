%matplotlib inline
import tensorflow as tf
from tensorflow import keras
import numpy as np
import plot_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import gan

BATCH_SIZE = 32
NUM_FEATURES = 100




#download data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0



# view some images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,1+i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i],cmap = plt.cm.binary)
plt.show()



import cv2
import matplotlib.pyplot as plt



#create and compile gan network
gan_net = gan.build_gan(NUM_FEATURES)
# train
x_train_dcgan = x_train.reshape(-1, 28, 28, 1) * 2. - 1.
dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
%%time
gan.train_dcgan(gan_net, dataset, BATCH_SIZE, NUM_FEATURES, epochs=10)

generator,discriminator = gan_net.layers
#generate_images
gan.create_and_save_single_image(generator,"gan_images/sample.jpg",BATCH_SIZE,NUM_FEATURES)
noise = tf.random.normal(shape = [BATCH_SIZE,NUM_FEATURES])
gen_images = generator(noise)
cv2.imwrite(gen_images[0,:,:,0],"prova.jpg")



