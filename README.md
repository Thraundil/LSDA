packages# LSDA
2018, DIKU - Cph University

# Group 4 (Emil, Frederik, Roberta, Steffen)


### Data saving and loading
* Download images, ~~resize to 299x299 (done in scraper)~~ and save as HDF5 files
* Figure out how to load HDF5 data into keras

### Model
* Use Inception-v3 as template.
* Training: Train dense layers first (save features outputtet by the conv-part, and train the dense layers only on that), then the whole architecture. Possibly implement boosting to extend rare classes. 
* Augment data on the fly: Follow Jupyter Notebook (CNNs_In_Practice_Data_Augmentation.ipynb) and possibly use this for extending rare classes
* Validate and create tensorboard graphs of accuracy
* Perform test predictions
* Implement tensorflow micro f1 score function as metric

# NOTES

### Downloading the data
Data can be downloaded with the scraper. To download all data (expected size: 25GB (with resizing)):  
`$ cd src`  
`$ python scraper.py`  
  
specific directories and the amount of files to download can also be specified. To download the first 500 images from the training dataset do:  
`$ cd src`  
`$ python scraper.py ../data/labels/train.json ../data/raw_images/train 500`

### Using the Framework
The framework is hopefully an easier way to access and keep track of our data, and try out classifiers. You can try out the current test version by:
`$ cd src`  
`$ python framework.py`  


### Kaggle Link:
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

### Team Name: K-NN or GTFO
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/team

### Overleaf Links
* Report 0: https://www.overleaf.com/15811080jqgcpbxscxdj#/60201263/
* Report 1: 

### Azure
login:
`lsda`
`CompetitionGroup4`

Connect with 
`ssh lsda@40.114.107.120`

Source into tensorflow enviroment:
`source lsdaenv/bin/activate`

# !!! DO NOT FORGET TO TURN OFF THE SERVER !!!

### Useful libraries
* CV2: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html#py-table-of-content-imgproc
* H5PY: http://docs.h5py.org/en/latest/quick.html


## Stuff
* Should the "missing"-images be deleted?
* Should (and can) we do the augmentation beforehand, i.e., not on the fly? Yes! https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
* Initializer for the on-top NN (https://keras.io/initializers/): Use either Glorot (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf, http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization) or He (https://arxiv.org/pdf/1502.01852.pdf), preferably uniform (https://github.com/keras-team/keras/issues/52). (Use He! Remember to avoid dead neurons in the first layer).
* Check if our pre-trained net uses ranges [0;255] or [0;1].
* Find a way to implement real-time (or almost) feedback during training. 
* Using a pre-trained model tutorials: https://keras.io/applications/, https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/.
* Remember to use the Keras function preprocess_input().
