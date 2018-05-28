import os
import cv2
import numpy as np
import time
import datetime
import h5py
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, AveragePooling2D, Dropout, Dense, Flatten
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# class Metrics(Callback):
#     def __init__(self):
#         super().__init__()
#         self.best_val_score_file = 'best_val_score.npz'
#         self.best_val_score = self.load_best_val_score()
#
#     def load_best_val_score(self):
#         if os.path.isfile(self.best_val_score_file):
#             return np.load('best_val_score.npz')[0]
#         else:
#             return 0
#
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(validation_features))).round()
#         val_targ = validation_labels
#         _val_f1 = f1_score(val_targ, val_predict, average='micro')
#         _val_recall = recall_score(val_targ, val_predict, average='micro')
#         _val_precision = precision_score(val_targ, val_predict, average='micro')
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
#         if _val_f1 > self.best_val_score:
#             print('Better model found, saving model with val score %s'%_val_f1)
#             self.model.save('best_full_model.h5')
#             self.best_val_score = _val_f1
#             # Save model
#             # Convert model to JSON and save
#             model_top_layer_json = model_top_layer.to_json()
#             with open('model_top_layer.json', 'w') as json_file:
#                 json_file.write(model_top_layer_json)
#             # Save weights
#             model_top_layer.save_weights('weights_top_layer.h5')
#             np.savez_compressed(self.best_val_score_file, np.array([_val_f1]))
#             print('Saved model as json and weights as h5..')
#         return
#
#
# class Save_model(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save('full_model.h5')
#         return


class TrainValTensorBoard(TensorBoard):
    """
    improved tensoroard logging, including validation set performance
    """
    def __init__(self, log_dir='./Tensorboard/', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        t = time.time()
        ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d_%H:%M:%S')
        training_log_dir = os.path.join(log_dir, ts+'_training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, ts+'_validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
