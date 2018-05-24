packages# LSDA
2018, DIKU - Cph University

# Group 4 (Emil, Frederik, Roberta, Steffen)


### Data loading
#### download images, ~~resize to 299x299~~ and save in h5 format

### Model
#### Use Inception-v3 as template.
#### Fine-tune the model: Train dense layers first (save features spat out by the conv-part and save them, and train the dense layers only on that), then the whole architecture. 
#### Load h5 data
#### Augment data on the fly: Follow Jupyter Notebook (CNNs_In_Practice_Data_Augmentation.ipynb) and use this for extending rare classes
#### validate and create tensorboard graphs of accuracy
#### perform test predictions

# NOTES

### Downloading the data
Data can be downloaded with the scraper. To download all data (expected time: 4h, expected size: 25GB (with resizing):  
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
#### https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

### Team Name: K-NN or GTFO
#### https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/team

### Overleaf Links
#### Report 0: https://www.overleaf.com/15811080jqgcpbxscxdj#/60201263/
#### Report 1: 

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
#### CV2: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html#py-table-of-content-imgproc
#### H5PY: http://docs.h5py.org/en/latest/quick.html


