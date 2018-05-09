from PIL import Image
import cv2

from src.constants import *
from src.feature_extraction.abstract_fature_extractor import AbstractFeatureExtractor


class ColorHistogramFeature(AbstractFeatureExtractor):

  def __init__(self, dataset='train'):
    super(ColorHistogramFeature, self).__init__(dataset)

    # prefix used to save files
    self.prefix = 'color_hist'

    # dimentionallity of the feature vector
    self.BINS = 4
    self.fature_dim = self.BINS ** 3
    return

  def calculate_feature(self, image: Image):
    """
    calculates the feature, returns the feature vector
    :param image: the PIL Image
    :return: vector of self.fature_dim components
    """
    bins = (self.BINS, self.BINS, self.BINS)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return hist.flatten()
