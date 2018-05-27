import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, AveragePooling2D, Dropout, Dense, Flatten
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

verbose = True

#===============================================================================
# IMPORT DATA
#===============================================================================
img_size = 299 # each RGB image has shape (img_size, img_size, 3) with values in (0;255)
no_labels = 228
no_imgs_batch = 100 # number of images loaded into memory

folder = '../data/raw_images/train/'
image_files = os.listdir(folder)
no_imgs_tot = len(image_files)

# Load and store features (x_train) in batches
x_train = np.zeros((no_imgs_batch, img_size, img_size, 3))
for i,image_file in enumerate(image_files):
    if i == no_imgs_batch:
        break
    image = load_img(os.path.join(folder,image_file), target_size=(img_size, img_size))
    image = img_to_array(image)
    x_train[i,:,:,:] = image
if verbose:
    print(f'Batch of size {no_imgs_batch} loaded..')

# Load and store labels (y_train)
annotations_dir = '../data/labels/annotations.npz'
y_train = np.load(annotations_dir)['arr_0'][1:no_imgs_batch+1,1:] # full annotations matrix is padded with one zero row and column, and has shape (no_imgs_tot+1,no_labels+1)

if verbose:
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)

#===============================================================================
# BUILD MODEL
#===============================================================================
model = Sequential()
incep_model = InceptionV3(weights="imagenet",
                          include_top=False,
                          input_shape=(img_size, img_size, 3))
model.add(incep_model)
# print('\n')
# for i,layer in enumerate(incep_model.layers):
#     print(f'Layer {i}:', layer)
# print('\n')
model.add(AveragePooling2D(pool_size=(8,8), strides=None, padding='valid', data_format=None))
model.add(Flatten())
model.add(Dense(no_labels, activation='sigmoid')) # top layer

if verbose:
    print('layer names: ', model.layers)
    print('dense_1 input_shape: ', model.get_layer('dense_1').input_shape)
    print('dense_1 output_shape: ', model.get_layer('dense_1').output_shape)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
if verbose:
    print('Model compiled..')
    print('Fitting model..')

#===============================================================================
# FIT MODEL
#===============================================================================
# model.fit(x_train, y_train)

# Structure data with datagenerator (w/o augmentation)
datagen = ImageDataGenerator(rescale=1./255)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=no_imgs_batch/32, epochs=2,
                    samples_per_epoch = no_imgs_batch)

# # Calculate predictions of the CNN
# bottleneck_features_train = model.predict_generator(datagen)
# # Save the output as a Numpy array
# np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
