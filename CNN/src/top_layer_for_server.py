import os
import cv2
import numpy as np
import time
import h5py
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, AveragePooling2D, Dropout, Dense, Flatten
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from utils import f1, TrainValTensorBoard

verbose = True

# %%============================================================================
# HYPERPARAMETERS
#===============================================================================
lr_Adam = 0.001
beta_1_Adam = 0.9
beta_2_Adam = 0.999
epsilon_Adam = 1e-8

lr_RMSProp = 0.001
lr_SGD = 0.01
momentum_SGD = 0.9

batch_size_top_layer = 1000
epochs_top_layer = 100

no_labels = 229 # NOTE: This is terrible!

# %%============================================================================
# IMPORT DATA FOR TRAINING TOP LAYER
#===============================================================================

# Load CNN features and corresponding labels NOTE: Labels are now of size 229 instead of 228!
os.chdir('/mnt/LSDA/CNN/src')
FEATURES_FOLDER = '/home/lsda/features'
LABELS_FOLDER = '/mnt/LSDA/labels'

# Load training CNN Features
DATASET = 'train'
f = h5py.File(os.path.join(FEATURES_FOLDER, DATASET, 'incept.hdf5'));
train_features = f['a'][1:]
f.close()
train_labels = np.load(os.path.join(LABELS_FOLDER, DATASET, 'labels.npz'))['arr_0'][1:]
if verbose:
    print('CNN features from training set shape: ', train_features.shape)
    print('CNN labels from training set shape: ', train_labels.shape, '\n')

# Load validation CNN features
DATASET = 'validation'
f = h5py.File(os.path.join(FEATURES_FOLDER, DATASET, 'incept.hdf5'));
validation_features = f['a'][1:]
f.close()
validation_labels = np.load(os.path.join(LABELS_FOLDER, DATASET, 'labels.npz'))['arr_0'][1:]

# %%============================================================================
# BUILD TOP LAYER
#===============================================================================
model_top_layer = Sequential()
model_top_layer.add(Dense(1024, # NOTE: Remove?
                activation='relu',
                kernel_initializer='he_uniform',
                input_shape=(2048,),
                name='dense_1'))
#model_top_layer.add(Dropout(0.5)) # NOTE: Remove?
model_top_layer.add(Dense(no_labels,
                kernel_initializer='he_uniform',
                activation='sigmoid',
                name='dense_2'))

# Choose and tune optimizer (Adam, RMSProp or SGD)
adam = optimizers.Adam(lr=lr_Adam,
                      beta_1=beta_1_Adam,
                      beta_2=beta_2_Adam,
                      epsilon=epsilon_Adam) # NOTE: epsilon=None is default
RMSProp = optimizers.RMSprop(lr=lr_RMSProp) # we should only tune lr
SGD = optimizers.SGD(lr=lr_SGD, momentum=momentum_SGD)

# Compile model
model_top_layer.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1])

# %%============================================================================
# TRAIN AND SAVE TOP LAYER
#===============================================================================
if verbose:
    print('Training top layer..')

# Callbacks
tbCallBack = TrainValTensorBoard(log_dir='./Tensorboard/',
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)
# Checkpoint
filepath = 'best_model_and_weights_top_layer_{val_f1:.2f}.h5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_f1',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

model_top_layer.fit(train_features, train_labels,
          epochs=epochs_top_layer,
          batch_size=batch_size_top_layer,
          callbacks=[checkpoint, tbCallBack],
          validation_data = (validation_features, validation_labels))
          # , validation_data=(validation_data, validation_labels)