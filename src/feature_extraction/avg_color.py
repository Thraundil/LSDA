import os
import numpy as np
from PIL import Image

from src.constants import *
from src.feature_extraction.abstract_fature_extractor import AbstractFeatureExtractor


class AvgColorFeature(AbstractFeatureExtractor):

  def __init__(self, dataset='train'):
    super(AvgColorFeature, self).__init__(dataset)

    # prefix used to save files
    self.prefix = 'avg_colors'

    # dimentionallity of the feature vector
    self.fature_dim = 3
    return

  def calculate_feature(self, image: Image):
    """
    calculates the feature, returns the feature vector
    :param image: the PIL Image
    :return: vector of self.fature_dim components
    """
    return np.mean(image, axis=(0, 1))
