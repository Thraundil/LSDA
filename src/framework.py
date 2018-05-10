import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from constants import LABEL_DIR, RAW_IMAGES_DIR, FEATURES_DIR, TOTAL_N_TRAIN
from feature_extraction.avg_color import AvgColorFeature
from feature_extraction.color_histogram import ColorHistogramFeature


def extract_annotations():
  file_path = os.path.join(LABEL_DIR,'train.json')
  assert os.path.exists(file_path), 'Error, could not find train.json in %s'%LABEL_DIR
  with open(file_path, 'r') as f:
    data = json.load(f)

  n_images = TOTAL_N_TRAIN 
  n_labels = max([max(map(int,img['labelId'])) for img in data['annotations']])
  annotations = np.zeros((n_images + 1,n_labels + 1)) #add 1 so we can access the row/column as the id directly
  for img in tqdm(data['annotations'],desc = 'Transforming annotations..'):
    for label in img['labelId']:
      annotations[int(img['imageId']),int(label)] = 1
  print('Saving annotations to %s file..'%os.path.join(LABEL_DIR,'annotations.npz'))
  np.savez_compressed(os.path.join(LABEL_DIR,'annotations'),annotations)


class Images:
  def __init__(self, n=10000, random_state=42, train_split=0.7, features=[]):
    print('Using %s random images from your %s directory, and splitting them into train and test sets'%(n, RAW_IMAGES_DIR))
    
    self.n = n
    self.random_state = random_state
    self.train_split = train_split

    # load annotations
    self._annotations = None
    # get list of shuffled image ids
    self.image_ids = self._get_image_ids(self.n, self.random_state)
    # split image ids
    self.train_ids, self.test_ids = train_test_split(self.image_ids, train_size=self.train_split, random_state=self.random_state)
    # save length of train and test sets
    self.n_train = len(self.train_ids)
    self.n_test = len(self.test_ids)

    # load features
    self.features = self._load_features(features)

    # turn feature matrix into a list of labels
    self.labels = self.to_label_list(self.annotations)

  def to_label_list(self, n_hot):
    """
    transforms a n-hot-matrix into a list of labels
    :return:
    """
    return np.array([np.where(labels == 1)[0] for labels in n_hot])

  @property
  def X_train(self):
    return self.features[self.train_ids]
  
  @property
  def X_test(self):
    return self.features[self.test_ids]
  
  @property
  def y_train(self):
    return self.annotations[self.train_ids]
  
  @property
  def y_test(self):
    return self.annotations[self.test_ids]
  
  @property
  def annotations(self):
    if self._annotations is None:
      self._annotations = self._load_annotations()
    return self._annotations
  
  def knn_or_gtfo(self, classifier = KNeighborsClassifier(2)):
    """
    runs the benchmark
    :param classifier:
    :return:
    """
    print('\n')
    start_time = pd.Timestamp.now()
    classifier.fit(self.X_train, self.y_train)
    print('Predicting on training set..')
    self.y_pred_train = classifier.predict(self.X_train)
    self.y_pred_train_f1 = self._eval_result(self.y_pred_train, self.y_train, plot=False)
    print('Predicting on test set..')
    self.y_pred_test = classifier.predict(self.X_test)
    self.y_pred_test_f1 = self._eval_result(self.y_test, self.y_pred_test, plot=True)
    print('Time taken: %s'%(pd.Timestamp.now() - start_time))

  def _eval_result(self, y_pred, y_true, plot=False):
    score = f1_score(y_pred, y_true, average='micro')
    print('Micro F1 Score: %s' % score)

    if plot:
      # determine F1-scores per label
      y_pred = y_pred.T
      y_true = y_true.T
      f1_per_feature = [f1_score(y_pred[i], y_true[i]) for i in range(1, 229)]
      feature_no = np.arange(1, 229)

      plt.figure(figsize=(6, 4.5))
      plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
      plt.bar(feature_no, f1_per_feature)
      plt.xlabel('Feature')
      plt.ylabel('F1 Score')
      plt.title('F1 score per feature')
      #plt.show()
      plt.savefig('../f1_per_feature.pdf')

    return score
    
  def _get_image_ids(self, n, random_state):
    """
    reads image ids and returns a list of shuffled ids
    :param n:
    :param random_state:
    :return:
    """
    print('Getting image ids..')
    image_files = os.listdir(os.path.join(RAW_IMAGES_DIR,'train'))
    assert len(image_files) > n, 'Error: trying to use n=%s when only %s images found in directory %s'%(n,len(image_files),RAW_IMAGES_DIR)
    np.random.seed(random_state)
    np.random.shuffle(image_files)
    return [int(image_file.split('.')[0]) for image_file in image_files[:n]]
  
  def _load_features(self, features):
    print('Loading Features: %s..'%', '.join([feature.__class__.__name__ for feature in features]))
    if len(features):
      # append feature matrices together
      feature_matrix = features[0].get_features(self.train_ids + self.test_ids)
      for feature in features[1:]:
        feature_matrix = np.append(feature_matrix, feature.get_features(self.train_ids + self.test_ids), axis=1)
    return feature_matrix
      
  def _load_annotations(self):
    print('Loading Annotations..')
    if not os.path.isfile(os.path.join(LABEL_DIR,'annotations.npz')):
      print('annotations.npz not found in %s, extracting from train.json'%LABEL_DIR)
      extract_annotations()
    return np.load(os.path.join(LABEL_DIR,'annotations.npz'))['arr_0']


if __name__ == '__main__':
  #example execution
  images = Images(n=11000, features=[AvgColorFeature(), ColorHistogramFeature()])
  images.knn_or_gtfo(classifier = KNeighborsClassifier(3))