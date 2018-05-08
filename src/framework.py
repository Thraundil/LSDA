import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from constants import LABEL_DIR, RAW_IMAGES_DIR, FEATURES_DIR, TOTAL_N_TRAIN

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
  
def create_avg_colors_npz():
  if not os.path.exists(os.path.join(FEATURES_DIR,'train')):
    os.makedirs(os.path.join(FEATURES_DIR,'train'))
  avg_colors = np.empty((TOTAL_N_TRAIN+1,3))
  avg_colors[:] = np.nan
  print('Creating average colors file: %s..'%os.path.join(FEATURES_DIR,'train','avg_colors'))
  np.savez_compressed(os.path.join(FEATURES_DIR,'train','avg_colors'),avg_colors)
  
def calc_and_save_avg_colors(image_ids):
  avg_colors = np.load(os.path.join(FEATURES_DIR,'train','avg_colors.npz'))['arr_0']
  for image_id in tqdm(image_ids,desc='Calculating Avg Colors..'):
    if np.isnan(avg_colors[image_id].sum()):
      avg_colors[image_id] = np.mean(plt.imread(os.path.join(RAW_IMAGES_DIR,'train',str(image_id)+'.jpg')), axis=(0,1))
     
  print('Saving updated average colors to %s..'%os.path.join(FEATURES_DIR,'train','avg_colors'))
  np.savez_compressed(os.path.join(FEATURES_DIR,'train','avg_colors'),avg_colors)
  return avg_colors

class Images():
  def __init__(self, n=10000, random_state=42, train_split=0.7, features=['avg_color']):
    print('Using %s random images from your %s directory, and splitting them into train and test sets'%(n, RAW_IMAGES_DIR))
    
    self.n = n
    self.random_state = random_state
    self.train_split = train_split
    
    self._annotations = None
    self.image_ids = self._get_image_ids(self.n, self.random_state)
    self.train_ids, self.test_ids = train_test_split(self.image_ids, train_size=self.train_split, random_state=self.random_state)
    self.n_train = len(self.train_ids)
    self.n_test = len(self.test_ids)
    
    self.features = self._load_features(features)
    
    self.labels = np.array([np.where(labels == 1)[0] for labels in self.annotations])

  @property
  def avg_color(self):
    if not '_avg_color' in self.__dict__:
      self._avg_color = self._load_avg_color()
    return self._avg_color
    
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
    print('\n')
    start_time = pd.Timestamp.now()
    classifier.fit(self.X_train, self.y_train)
    print('Predicting on training set..')
    self.y_pred_train = classifier.predict(self.X_train)
    self.y_pred_train_f1 = f1_score(self.y_train, self.y_pred_train, average='micro')
    print('Micro F1 Score, train: %s'%self.y_pred_train_f1)
    print('Predicting on test set..')
    self.y_pred_test = classifier.predict(self.X_test)
    self.y_pred_test_f1 = f1_score(self.y_test, self.y_pred_test, average='micro')
    print('Micro F1 Score, test: %s'%self.y_pred_test_f1)
    print('Time taken: %s'%(pd.Timestamp.now() - start_time))
    
  def _get_image_ids(self, n, random_state):
    print('Getting image ids..')
    image_files = os.listdir(os.path.join(RAW_IMAGES_DIR,'train'))
    assert len(image_files) > n, 'Error: trying to use n=%s when only %s images found in directory %s'%(n,len(image_files),RAW_IMAGES_DIR)
    np.random.seed(random_state)
    np.random.shuffle(image_files)
    return [int(image_file.split('.')[0]) for image_file in image_files[:n]]
    
  def _load_avg_color(self):
    print('Loading Average Colors..')
    if not os.path.isfile(os.path.join(FEATURES_DIR,'train','avg_colors.npz')):
      create_avg_colors_npz()
    avg_colors = calc_and_save_avg_colors(self.train_ids+self.test_ids)
    return avg_colors
  
  def _load_features(self, features):
    print('Loading Features: %s..'%', '.join(features))
    for feature in features:
      assert hasattr(self,feature), 'Error, could not find feature %s in class'%feature
    if len(features):
      feature_matrix = self.__getattribute__(features[0])
      for feature in features[1:]:
        feature_matrix = np.append(feature_matrix, self.__getattribute__(feature),axis=1)
    return feature_matrix
      
  def _load_annotations(self):
    print('Loading Annotations..')
    if not os.path.isfile(os.path.join(LABEL_DIR,'annotations.npz')):
      print('annotations.npz not found in %s, extracting from train.json'%LABEL_DIR)
      extract_annotations()
    return np.load(os.path.join(LABEL_DIR,'annotations.npz'))['arr_0']

if __name__ == '__main__':
  #example execution
  images = Images(n=15000, features=['avg_color'])
  images.knn_or_gtfo(classifier = KNeighborsClassifier(3))