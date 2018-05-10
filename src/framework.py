import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from constants import LABEL_DIR, RAW_IMAGES_DIR, FEATURES_DIR, TOTAL_N, SUBMISSION_FILE_DIR
from feature_extraction.avg_color import AvgColorFeature
from feature_extraction.color_histogram import ColorHistogramFeature


def extract_annotations(dataset='train'):
  file_path = os.path.join(LABEL_DIR,'train.json')
  assert os.path.exists(file_path), 'Error, could not find train.json in %s'%LABEL_DIR
  with open(file_path, 'r') as f:
    data = json.load(f)

  n_images = TOTAL_N[dataset]
  n_labels = max([max(map(int,img['labelId'])) for img in data['annotations']])
  annotations = np.zeros((n_images + 1,n_labels + 1)) #add 1 so we can access the row/column as the id directly
  for img in tqdm(data['annotations'],desc = 'Transforming annotations..'):
    for label in img['labelId']:
      annotations[int(img['imageId']),int(label)] = 1
  print('Saving annotations to %s file..'%os.path.join(LABEL_DIR,'annotations.npz'))
  np.savez_compressed(os.path.join(LABEL_DIR,'annotations'),annotations)


class Images:
  def __init__(self, n=10000, random_state=42, train_split=0.7, feature_classes=[]):
    print('Using %s random images from your %s directory, and splitting them into train and test sets'%(n, RAW_IMAGES_DIR))
    
    self.n = n
    self.train_split = train_split
    self.random_state = random_state
    self.feature_classes = feature_classes
    

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
    self.train_features = self._load_features(self.feature_classes, self.image_ids)

    # turn feature matrix into a list of labels
    self.labels = np.array([np.where(rows == 1)[0] for rows in self.annotations])

    
  @property
  def X_train(self):
    return self.train_features[self.train_ids]
  
  @property
  def X_test(self):
    return self.train_features[self.test_ids]
  
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
    self.y_pred_train_f1 = f1_score(self.y_train, self.y_pred_train, average='micro')
    print('Micro F1 Score, train: %s' % self.y_pred_train_f1)
    print('Predicting on test set..')
    self.y_pred_test = classifier.predict(self.X_test)
    self.y_pred_test_f1 = f1_score(self.y_test, self.y_pred_test, average='micro')
    print('Micro F1 Score, test: %s'%self.y_pred_test_f1)
    print('Time taken: %s'%(pd.Timestamp.now() - start_time))
    
  def run_on_test_and_make_kaggle_sub_file(self, classifier = KNeighborsClassifier(3)):
    test_feature_classes = [feature_class.__class__(dataset='test') for feature_class in self.feature_classes]
    test_image_ids = range(1,TOTAL_N['test']+1)
    test_features = self._load_features(test_feature_classes, test_image_ids)
    print('\n')
    start_time = pd.Timestamp.now()
    classifier.fit(self.X_train, self.y_train)
    print('Predicting on training set..')
    self.y_pred_train = classifier.predict(self.X_train)
    self.y_pred_train_f1 = f1_score(self.y_train, self.y_pred_train, average='micro')
    print('Micro F1 Score, train: %s' % self.y_pred_train_f1)
    print('Predicting on test set..')
    self.y_pred_test = classifier.predict(test_features[test_image_ids])
    print('Time taken: %s'%(pd.Timestamp.now() - start_time))
    
    print('Creating submission file in %s..'%SUBMISSION_FILE_DIR)
    if not os.path.exists(SUBMISSION_FILE_DIR):
      os.makedirs(SUBMISSION_FILE_DIR)
    labels = np.array([np.where(rows == 1)[0] for rows in self.y_pred_test])
    out = np.c_[test_image_ids,labels]
    out = [','.join([str(row[0]), ' '.join(list(map(str,row[1])))]) for row in out]
    save_time = pd.Timestamp.now()
    file_name = save_time.strftime('%Y%m%d_%H%M') + '_submission.csv'
    with open(os.path.join(SUBMISSION_FILE_DIR,file_name), 'w+') as f:
      f.write('image_id,label_id\n')
      for row in out:
        f.write(row+'\n')
    print('File Saved')
    
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
  
  def _load_features(self, features, img_ids):
    print('Loading Features: %s..'%', '.join([feature.__class__.__name__ for feature in features]))
    if len(features):
      # append feature matrices together
      feature_matrix = features[0].get_features(img_ids)
      for feature in features[1:]:
        feature_matrix = np.append(feature_matrix, feature.get_features(img_ids), axis=1)
    return feature_matrix
      
  def _load_annotations(self):
    print('Loading Annotations..')
    if not os.path.isfile(os.path.join(LABEL_DIR,'annotations.npz')):
      print('annotations.npz not found in %s, extracting from train.json'%LABEL_DIR)
      extract_annotations()
    return np.load(os.path.join(LABEL_DIR,'annotations.npz'))['arr_0']