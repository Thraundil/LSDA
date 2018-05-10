import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List
from PIL import Image

from constants import *

class AbstractFeatureExtractor():
  """
  An abstract class used to implement feature extraction
  """

  def __init__(self, dataset='train'):
    # prefix used to save files
    self.prefix = None

    # dimentionallity of the feature vector
    self.fature_dim = None

    # dataset to target
    self.dataset = dataset
    return

  def calculate_feature(self, image: Image):
    """
    calculates the feature, returns the feature vector
    :param image: the PIL Image
    :return: vector of self.fature_dim components
    """

    ## IMPLEMENT THIS IN THE CONCRETE CLASS

    pass

  def get_features(self, img_ids):
    """
    returns a Matrix of features for the training set. PRecomputing and saving this matrix is advised.
    :return:
    """
    print('Loading Average Colors..')
    if not os.path.isfile(os.path.join(FEATURES_DIR, self.dataset, self.prefix + '.npz')):
      self.create_blank_file()
    avg_colors = self.load_features(img_ids)
    return avg_colors

  def create_blank_file(self):
    """
    initializes a blank file
    :return:
    """
    if not os.path.exists(os.path.join(FEATURES_DIR, self.dataset)):
      os.makedirs(os.path.join(FEATURES_DIR, self.dataset))
    avg_colors = np.empty((TOTAL_N_TRAIN + 1, self.fature_dim))
    avg_colors[:] = np.nan
    print('Creating blank file for feature %s as %s..' % (self.prefix, os.path.join(FEATURES_DIR, self.dataset, self.prefix)))
    np.savez_compressed(os.path.join(FEATURES_DIR, self.dataset, self.prefix), avg_colors)
    return

  def load_features(self, image_ids: List[int]):
    """
    loads the features, calculates unknown ones
    :param image_ids: ids of images to load
    :return:
    """
    features = np.load(os.path.join(FEATURES_DIR, self.dataset, self.prefix + '.npz'))['arr_0']
    for image_id in tqdm(image_ids, desc=('Calculating ' + self.prefix + '..')):
      if np.isnan(features[image_id].sum()):
        img = plt.imread(os.path.join(RAW_IMAGES_DIR, self.dataset, str(image_id) + '.jpg'))
        features[image_id] = self.calculate_feature(img)

    # save the updated features
    print('\nSaving updated %s to %s..' % (self.prefix, os.path.join(FEATURES_DIR, self.dataset, self.prefix)))
    np.savez_compressed(os.path.join(FEATURES_DIR, self.dataset, self.prefix), features)
    return features