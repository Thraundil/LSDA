import numpy as np
import cv2

from constants import *
from feature_extraction.abstract_feature_extractor import AbstractFeatureExtractor


class GreyScaleImg(AbstractFeatureExtractor):

  def __init__(self, dataset='train'):
    super(GreyScaleImg, self).__init__(dataset)

    # prefix used to save files
    self.prefix = 'grey_scale_img'

    # dimensionality of the feature vector (e.g. 144 for 12x12 img)
    self.size = 12
    self.feature_dim = self.size*self.size
    return

  def calculate_feature(self, image):
    """
    calculates the feature, returns the feature vector
    :param image: the PIL Image
    :return: vector of self.fature_dim components
    """
    image = np.array(image)
    image = cv2.resize(image, (self.size,self.size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).flatten()
    
    return image
