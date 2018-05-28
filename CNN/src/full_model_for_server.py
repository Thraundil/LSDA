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
from keras.callbacks import Callback, TensorBoard
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from utils import f1, Metrics, Save_model

verbose = True
save_model = Save_model()

# %%============================================================================
# HYPERPARAMETERS
#===============================================================================
no_imgs_batch = 4000 # number of images loaded into memory. The "meta-batch"
lr_SGD = 1e-4 # suggested very low lr
momentum_SGD = 0.9
batch_size = 500 # batch size given to fit function
epochs = 100

# %%============================================================================
# IMPORT DATA FOR FINE-TUNING FULL MODEL
#===============================================================================
# Features
train_dir = '../data/raw_images/train/'
image_files = os.listdir(train_dir)
image_files = np.random.permutation(image_files)
no_imgs_tot = len(image_files)
img_size = 299 # each RGB image has shape (img_size, img_size, 3) with values in (0;255)

# Labels
annotations_dir = '../data/labels/train/labels.npz'
train_labels = np.load(annotations_dir)['arr_0']
no_labels = 228

for iteration in range(int(no_imgs_tot/no_imgs_batch+1)):
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

    if verbose:
        print('x_train shape: ', x_train.shape)
        print('y_train shape: ', y_train.shape)

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
json_file = open('model_top_layer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_top_layer = model_from_json(loaded_model_json)
model_top_layer.load_weights('weights_top_layer.h5')
print('Loaded top layer from disk')

# Add top layer to full model
model_full.add(model_top_layer)

# Choose and tune optimizer
SGD_full = optimizers.SGD(lr=lr_SGD, momentum=momentum_SGD)

# Compile model
model_full.compile(optimizer=SGD_full,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

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
# FINE-TUNE FULL MODEL
#===============================================================================
if verbose:
    print('Fine-tuning full model..')

# Structure data with datagenerator (with augmentation)
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True)

model_full.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         epochs=epochs,callbacks=[save_model])
                         # steps_per_epoch=no_imgs_batch/32, samples_per_epoch = no_imgs_batch
