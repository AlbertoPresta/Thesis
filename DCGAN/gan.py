
import tensorflow as tf
from tensorflow import keras
import numpy as np
import plot_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import joblib



def build_gan(num_features):
    generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[num_features]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, (5,5), (2,2), padding="same", activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, (5,5), (2,2), padding="same", activation="tanh"),
    ])

    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, (5,5), (2,2), padding="same", input_shape=[28, 28, 1]),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (5,5), (2,2), padding="same"),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan = keras.models.Sequential([generator, discriminator])
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return gan




def train_dcgan(gan, dataset, batch_size, num_features, epochs=5,nome = "model/gan.sav"):
    generator, discriminator = gan.layers
    seed = tf.random.normal(shape=[batch_size, num_features])
    for epoch in tqdm(range(epochs)):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape=[batch_size, num_features])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)
    #joblib.dump(gan,"model/gan.sav")


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(10,10))

  for i in range(25):
      plt.subplot(5, 5, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
      plt.axis('off')

  plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()






  #generate_images
  def create_and_save_single_image(generator,save_pt,batch_size,num_features):
      noise = tf.random.normal(shape = [batch_size,num_features])
      gen_images = generator(noise)
      cv2.imwrite(save_pt)
