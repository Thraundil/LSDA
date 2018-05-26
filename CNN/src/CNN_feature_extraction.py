
# coding: utf-8

# # CNN feature extraction
# extracts features from images using a pretrained cnn

# In[ ]:


import os
import cv2
import numpy as np
import h5py
import time

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import AveragePooling2D, Flatten
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


BATCH_SIZE = 500


# In[ ]:


MODEL = Sequential()
MODEL.add(InceptionV3(weights="imagenet", include_top=False))
MODEL.add(AveragePooling2D(pool_size=(8,8), strides=None, padding='valid', data_format=None))


# In[ ]:


def load_images(raw_images_dir, start, end):
    images = np.zeros((end - start, 299, 299, 3))
    for i in range(start, end):
        image = load_img(os.path.join(raw_images_dir, "{}.jpg".format(i)) , target_size= (299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        images[i- start] = image[0]
    return images
        


# In[ ]:


def extract(raw_images_dir, out_dir):
    # get list of images
    image_files = os.listdir(raw_images_dir)
    
    n = len(image_files)
    
    features = np.zeros((n+1, 2048))
    
    with tqdm(range(0, n)) as progress_bar:
        batch_start = 1
        while batch_start < n:
            this_batch_size = min(BATCH_SIZE, n - batch_start + 1)
            upper_bound = min(batch_start + BATCH_SIZE, n+1)
            
            # get images
            batch = load_images(raw_images_dir, batch_start, upper_bound)
            
            # get ccn predictions
            preds = MODEL.predict(batch).reshape([this_batch_size, 2048])
            
            # save to features
            features[batch_start:upper_bound] = preds
            
            batch_start += this_batch_size
            progress_bar.update(this_batch_size)
        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    f = h5py.File(out_dir + 'incept.hdf5')
    f['a'] = features
    f.close()
    


# In[ ]:


if not os.path.exists('../data/features/'):
    os.mkdir('../data/features/')
extract('../data/raw_images/validation/', '../data/features/validation/')
extract('../data/raw_images/train/', '../data/features/train/')
extract('../data/raw_images/test/', '../data/features/test/')

