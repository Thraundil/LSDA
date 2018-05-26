import os
import cv2
import numpy as np
from tqdm import tqdm
from itertools import chain, repeat, cycle

from keras.models import Sequential
from keras.layers import Activation, AveragePooling2D, Dropout, Dense, Flatten
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#===============================================================================
# IMPORT DATA
#===============================================================================

img_size = 299
no_imgs = 100
no_labels = 228

folder = '../data/raw_images/train/'
image_files = os.listdir(folder)
no_imgs_tot = len(image_files)

# Load and store features in batches
x_train = np.zeros((no_imgs, img_size, img_size, 3))
for i,image_file in enumerate(image_files):
    if i == no_imgs:
        break
    image = load_img(os.path.join(folder,image_file), target_size=(img_size,img_size))
    image = img_to_array(image)
    x_train[i,:,:,:] = image
print(f'Batch of size {no_imgs} loaded..')

# Load and store labels
# annotations_dir = '../data/labels/annotations.npz'
# y_train = np.load(annotations_dir)['arr_0'][1:no_imgs+1,1:] # full annotations matrix is padded with one zero row and column, and has shape (no_imgs_tot+1,no_labels+1)
y_train = np.zeros((no_imgs,no_labels)) # THIS IS JUST A TEST CONSTRUCTION!
for i in range(y_train.shape[0]):
    r = np.random.randint(no_labels)
    y_train[i,r] = 1
no_labels = y_train.shape[1]
print(np.sum(y_train))

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)

# Structure in datagenerator (w/o augmentation)
datagen = ImageDataGenerator(rescale=1./255)

#===============================================================================
# DEFINE MODEL
#===============================================================================

# InceptionV3
incep_model = InceptionV3(weights="imagenet", include_top=False)
model = Sequential()
model.add(incep_model)
model.add(AveragePooling2D(pool_size=(8,8), strides=None, padding='valid', data_format=None))
# Top layer
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(no_labels, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Model compiled..')

model.fit(x_train, y_train)
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=no_imgs/32, epochs=2)

# # Calculate predictions of the CNN
# bottleneck_features_train = model.predict_generator(datagen)
# # Save the output as a Numpy array
# np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
