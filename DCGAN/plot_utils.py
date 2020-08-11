import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_images(model,epochs,test_input):
    predictions = model(test_input,training = False)
    fig = plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5 + 127.5,cmap = 'binary')
        plt.axis('off')
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
