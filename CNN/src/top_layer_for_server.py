import os
import cv2
import numpy as np
import time
import h5py
from tqdm import tqdm

import tensorflow as tf
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
# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('opt_choice', 'adam', 'Choice of optimizer. Valid input are "adam", "RMSProp" and "SGD"')
flags.DEFINE_float('lr_Adam', 0.001, 'Adam learning rate')
flags.DEFINE_float('beta_1_Adam', 0.9, 'Adam beta_1')
flags.DEFINE_float('beta_2_Adam', 0.999, 'Adam beta_2')
flags.DEFINE_float('epsilon_Adam', 1e-8, 'Adam epsilon')
flags.DEFINE_float('lr_RMSProp', 0.001, 'RMSProp learning rate')
flags.DEFINE_float('lr_SGD', 0.01, 'SGD learning rate')
flags.DEFINE_float('momentum_SGD', 0.9, 'SGD momentum')
flags.DEFINE_integer('batch_size', 1000, 'batch size')
flags.DEFINE_integer('epochs', 100, 'epochs')

opt_choice = FLAGS.opt_choice
lr_Adam = FLAGS.lr_Adam
print(lr_Adam)
beta_1_Adam = FLAGS.beta_1_Adam
beta_2_Adam = FLAGS.beta_2_Adam
epsilon_Adam = FLAGS.epsilon_Adam
lr_RMSProp = FLAGS.lr_RMSProp
lr_SGD = FLAGS.lr_SGD
momentum_SGD = FLAGS.momentum_SGD
batch_size = FLAGS.batch_size
epochs = FLAGS.epochs

no_labels = 229 # NOTE: This is actually no_labels+1 and is objectively terrible!

print(opt_choice)
assert(1 == 0)

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
if opt_choice == 'adam':
    opt = optimizers.Adam(lr=lr_Adam,
                          beta_1=beta_1_Adam,
                          beta_2=beta_2_Adam,
                          epsilon=epsilon_Adam) # NOTE: epsilon=None is default
elif opt_choise == 'RMSProp':
    opt = optimizers.RMSprop(lr=lr_RMSProp) # we should only tune lr
else opt_choise == 'SGD':
    opt = optimizers.SGD(lr=lr_SGD, momentum=momentum_SGD)

# Compile model
model_top_layer.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1])

# %%============================================================================
# TRAIN AND SAVE TOP LAYER
#===============================================================================
if verbose:
    print('Training top layer..')

# Callbacks
tbCallBack = TrainValTensorBoard(log_dir='./Tensorboard/top_layer/',
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
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[checkpoint, tbCallBack],
          validation_data = (validation_features, validation_labels))
          # , validation_data=(validation_data, validation_labels)
