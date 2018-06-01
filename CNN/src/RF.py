import os
import sys
import numpy as np
import time
import h5py
import tensorflow as tf
from functools import reduce
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('verbose', True, 'Boolean. Whether or not to show verbose output.')
flags.DEFINE_float('sub_sample_frac', 0.1, 'If there are n_samples_total in total, then each RF will be trained on n_samples_total*sub_sample_frac randomly chosen samples. The reason for this subsampling is that RFs scale as n*log(n) and that memory issues are otherwise encountered.')
flags.DEFINE_integer('n_RFs', 10, 'Number of RFs that are built on the subsamples.')
flags.DEFINE_integer('n_estimators', 10, 'Number of trees in each sub RF.')
flags.DEFINE_integer('max_depth', 30, 'Maximum tree depth for each sub RF.')

verbose = FLAGS.verbose
sub_sample_frac = FLAGS.sub_sample_frac
n_RFs = FLAGS.n_RFs
n_estimators = FLAGS.n_estimators
max_depth = FLAGS.max_depth

# %%============================================================================
# IMPORT DATA
#===============================================================================
if verbose:
    print('Loading data..')

# Load CNN features and corresponding labels
os.chdir('/mnt/LSDA/CNN/src')
FEATURES_FOLDER = '/home/lsda/features'
LABELS_FOLDER = '/mnt/LSDA/labels'

# Load training CNN features
DATASET = 'train'
f = h5py.File(os.path.join(FEATURES_FOLDER, DATASET, 'incept.hdf5'));
x_train = f['a'][1:,1:]
f.close()
y_train = np.load(os.path.join(LABELS_FOLDER, DATASET, 'labels.npz'))['arr_0'][1:,1:] # NOTE: Labels are now of size 229 instead of 228!
if verbose:
    print('CNN features from training set shape: ', x_train.shape)
    print('CNN labels from training set shape: ', y_train.shape, '\n')

# Load validation CNN features
DATASET = 'validation'
f = h5py.File(os.path.join(FEATURES_FOLDER, DATASET, 'incept.hdf5'));
x_val = f['a'][1:1,:]
f.close()
y_val = np.load(os.path.join(LABELS_FOLDER, DATASET, 'labels.npz'))['arr_0'][1:,1:]

# %%============================================================================
# TRAIN RANDOM FORESTS ON SUBSETS OF FEATURES, THEN COMBINE THEM
#===============================================================================
def generate_rf(x_train, y_train, x_val, y_val, n_estimators, max_depth, n_generated_RFs, verbose):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features='log2', n_jobs=-1)
    rf.fit(x_train, y_train)
    if verbose:
        print('f1 score for RF number {}: {}'.format(n_generated_RFs, f1_score(y_val, rf.predict(x_val), average='micro')))
    return rf

def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

rfs = [] # collects all RFs

if verbose:
    print('Training {} RFs with n_estimators = {} and max_depth = {} on {} random subsamples each'.format(n_RFs, n_estimators, max_depth, y_train.shape[0]))

t1 = time.time()
for n_generated_RFs in range(n_RFs):
    # Generate indices for subsample of data to be trained on.
    # Use StratifiedShuffleSplit to get all labels represented.
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sub_sample_frac)
    indices,_ = sss.split(x_train, y_train) # gives the training indices
    print(len(indices))

    # Train the RF
    rf = generate_rf(x_train[indices,:], y_train[indices,:], x_val, y_val, n_estimators, max_depth, n_generated_RFs, verbose)
    rfs.append(rf)
    del indices,_,rf # free some memory
t2 = time.time()
print('Total training time: {} seconds'.format(t2-t1))

# Combine the list of random forest models into one giant forest
rf_combined = reduce(combine_rfs, rfs)
print('f1 score from combined RF: {}'.format(f1_score(y_val, rf_combined.predict(x_val), average='micro')))

# save model
joblib.dump(rf_combined, 'RF_trained.pkl')

# For loading in prediction script, do:
# clf = joblib.load('RF_trained.pkl')

# A kind of a grid-search is made over the parameter values seen below. After this
# is done, this script is run again, but with the best (also according to runtime)
# n_estimators and max_depth for every RF, which will then be run multiple times
# to create a lot of similar RFs, which are then combined into rf_combined, which
# is the final model.
# n_estimators_list = np.arange(10,121,30)
# max_depth_list = np.arange(20,61,10)
# Use the following as the for loop instead:
# for n_estimators in n_estimators_list:
    # for max_depth in max_depth_list:
#
# if verbose:
#     print('RFs with the following parameters will be trained on subsamples:')
#     print('n_estimators: ', n_estimators_list)
#     print('max_depth: ', max_depth_list)
