import os
import cv2
import numpy as np
import time
import h5py
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras import backend as K
from keras import optimizers
from keras.models import Sequential, load_model, model_from_json
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
no_imgs_batch = 4000 # number of images loaded into memory. The "meta-batch"
lr_SGD = 1e-4 # suggested very low lr
momentum_SGD = 0.9

batch_size = 500 # batch size given to fit function
epochs = 2

# %%============================================================================
# IMPORT VALIDATION DATA
#===============================================================================
#Load subset of validation data to check on
N_VALID = 1000

val_dir = '../data/raw_images/validation/'
val_image_files = os.listdir(val_dir)
val_image_files = np.random.permutation(val_image_files)[:N_VALID]

val_indices = [int(image_file.split('.')[0]) for image_file in val_image_files]
val_annotations_dir = '../data/labels/validation/labels.npz'
y_val = np.load(val_annotations_dir)['arr_0'][val_indices]

x_val = np.zeros((N_VALID, img_size, img_size, 3))
for i,image_file in enumerate(val_image_files):
    try:
        image = load_img(os.path.join(val_dir,image_file), target_size=(img_size, img_size))
        image = img_to_array(image)
        x_val[i,:,:,:] = image
    except:
        print('Failed to load image %s'%image_file)
if verbose:
    print('x_val shape: ', x_val.shape)
    print('y_val shape: ', y_val.shape)

# %%============================================================================
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
if verbose:
    print('Last frozen layer is %s'%last_frozen_layer)
for layer in incep_model.layers[:last_frozen_layer]:
    layer.trainable = False
for layer in incep_model.layers[last_frozen_layer:]:
    layer.trainable = True

model_full.add(incep_model)
model_full.add(AveragePooling2D(pool_size=(8,8)))
model_full.add(Flatten())

# Load top layer with weights
model_top_layer = load_model('best_model_and_weights_top_layer_0.48.h5',custom_objects = {'f1':f1})

# Add top layer to full model
model_full.add(model_top_layer)

# Choose and tune optimizer
SGD_full = optimizers.SGD(lr=lr_SGD, momentum=momentum_SGD)

# Compile model
model_full.compile(optimizer=SGD_full,
                   loss='categorical_crossentropy',
                   metrics=['accuracy',f1])

# if verbose:
#     print('')
#     print('Full model compiled.')
#     print('Here are some details regarding the architecture:')
#     for i,chunk in enumerate(model_full.layers):
#         if i == 0: # unpack CNN
#             print('CNN layers:')
#             print('First CNN layer input shape: ', chunk.layers[0].input_shape)
#             print('...')
#             print('Last CNN layer output shape: ', chunk.layers[-1].output_shape, '\n')
#         if i == 1: # Flatten layer
#             print('Flatten layer:')
#             print('Flatten layer input shape: ', chunk.input_shape)
#             print('Flatten layer output shape: ', chunk.output_shape, '\n')
#         if i == 2: # AveragePooling2D
#             print('Average pooling layer:')
#             print('AvPool layer input shape: ', chunk.input_shape)
#             print('AvPool layer output shape: ', chunk.output_shape, '\n')
#         if i == 3: # unpack top_layer
#             print('Top layers:')
#             print('First top_layer layer input shape: ', chunk.layers[0].input_shape)
#             print('...')
#             print('Last top_layer layer output shape: ', chunk.layers[-1].output_shape, '\n')
#             # for j,layer in enumerate(chunk.layers):
#             #     print(f'Layer {j} input shape: ', layer.input_shape)
#             #     print(f'Layer {j} output shape: ', layer.output_shape)
#     # from keras.utils import plot_model
#     # plot_model(model_full, to_file='full_model.png', show_shapes=True) # spits out a flow-chart of the model

# %%============================================================================
# FINE-TUNE FULL MODEL IN BATCHES OF LOADED DATA
#===============================================================================
if verbose:
    print('Fine-tuning full model..')

#$$$$$$$$$$$$$$$$$$$$$ Arguments passed to fit_generator $$$$$$$$$$$$$$$$$$$$$$$
# Structure data with datagenerator (with augmentation)
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True)

# Callbacks
tbCallBack = TrainValTensorBoard(log_dir='./Tensorboard/full_model/',
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)
# Checkpoint
filepath = 'best_model_and_weights_full_model.h5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_f1',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$ Prepare to load data $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# x_train (features)
train_dir = '../data/raw_images/train/'
image_files = os.listdir(train_dir)
image_files = np.random.permutation(image_files)
no_imgs_tot = len(image_files)
img_size = 299 # each RGB image has shape (img_size, img_size, 3) with values in (0;255)

# y_train (labels)
annotations_dir = '../data/labels/train/labels.npz'
train_labels = np.load(annotations_dir)['arr_0']

for iteration in range(int(no_imgs_tot/no_imgs_batch+1)):
    # Load data
    print('Iteration: %s'%iteration)
    subset = image_files[(iteration*no_imgs_batch):((iteration+1)*no_imgs_batch)]
    indices = [int(image_file.split('.')[0]) for image_file in subset]

    # Load and store features (x_train) in batches
    x_train = np.zeros((no_imgs_batch, img_size, img_size, 3))
    for i,image_file in enumerate(subset):
        try:
            image = load_img(os.path.join(train_dir,image_file), target_size=(img_size, img_size)) # NOTE: Speed up using multithreading!
            image = img_to_array(image)
            x_train[i,:,:,:] = image
        except:
            print('Failed to load image %s'%image_file)

    # Load and store labels (y_train)
    y_train = train_labels[indices] # full annotations matrix is padded with one zero row and column, and has shape (no_imgs_tot+1,no_labels+1)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Perform the fit $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    model_full.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                             epochs=epochs,
                             callbacks=[checkpoint, tbCallBack],
                             validation_data = (x_val, y_val))
                             # steps_per_epoch=no_imgs_batch/32, samples_per_epoch = no_imgs_batch
