import os
import cv2
import numpy as np
import h5py

from keras.models import Sequential
from keras.layers import Activation, AveragePooling2D, Dropout, Dense, Flatten
from keras import optimizers
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

train_dir = '../data/raw_images/train/'
image_files = os.listdir(train_dir)
no_imgs_tot = len(image_files)

# Load and store features (x_train) in batches
x_train = np.zeros((no_imgs_batch, img_size, img_size, 3))
for i,image_file in enumerate(image_files):
    if i == no_imgs_batch:
        break
    image = load_img(os.path.join(train_dir,image_file), target_size=(img_size, img_size))
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
# BUILD TOP LAYER
#===============================================================================
# Load files
# NOTE that these files may be too large to fit in memory!

# Load features extracted from the CNN
# f = h5py.File('filename')
# CNN_features = f['a']
# f.close()
CNN_features = np.random.rand(no_imgs_batch,2048) # test construction
# train_data = np.load(open('bottleneck_features_train.npy'))
if verbose:
    print('CNN_features shape: ', CNN_features.shape)

# validation_data = ...
# validation_labels = ...

model_top_layer = Sequential()
# model_top_layer.add(Flatten(input_shape=(1,2048))) # the comma in (2048,) is important!
model_top_layer.add(Dense(1024, activation='relu', input_shape=(2048,)))
model_top_layer.add(Dropout(0.5))
model_top_layer.add(Dense(no_labels,
                kernel_initializer='he_uniform',
                activation='sigmoid'))
model_top_layer.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if verbose:
    print('Training top layer..')

#===============================================================================
# TRAIN TOP LAYER
#===============================================================================
model_top_layer.fit(CNN_features, y_train,
          epochs=50,
          batch_size=32) # , validation_data=(validation_data, validation_labels)
# model_top_layer.save_weights('top_layer.h5')

#===============================================================================
# BUILD FULL MODEL
#===============================================================================
model_full = Sequential()

# Load and freeze (all but last few) InceptionV3 CNN layers
incep_model = InceptionV3(weights="imagenet",
                          include_top=False,
                          input_shape=(img_size, img_size, 3))
# Freeze all but the last inception modules (https://arxiv.org/pdf/1409.4842.pdf)
# To see list of layers, run:
# for i,layer in enumerate(incep_model.layers):
#     print(f'Layer {i}:', layer)
# Last and second to last layers begins in layer 280 and 249, respectively (using 1-indexing)
last_frozen_layer = 279
for layer in incep_model.layers[:last_frozen_layer]:
    layer.trainable = False
for layer in incep_model.layers[last_frozen_layer:]:
    layer.trainable = True

model_full.add(incep_model)
model_full.add(AveragePooling2D(pool_size=(8,8)))
# model_top_layer.load_weights('model_top_layer_weights_path')
model_full.add(model_top_layer)

model_full.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# NOTE: Begin debugging here!
# if verbose:
    #print('layer names: ', model.layers)
    # print('flatten_1 input_shape: ', model_top_layer.get_layer('flatten_1').input_shape)
    # print('flatten_1 output_shape: ', model_top_layer.get_layer('flatten_1').output_shape)


#===============================================================================
# FINE-TUNE FULL MODEL
#===============================================================================
# Structure data with datagenerator (with augmentation)
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True)

model_full.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=no_imgs_batch/32, epochs=2,
                    samples_per_epoch = no_imgs_batch)
