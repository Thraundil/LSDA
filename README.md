packages# LSDA
2018, DIKU - Cph University

# Group 4 (Emil, Frederik, Roberta, Steffen)

# NOTES

### Downloading the data
Data can be downloaded with the scraper. To download all data (expected time: 4h, expected size: 100GB):  
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
#### Rapport 0: https://www.overleaf.com/15811080jqgcpbxscxdj#/60201263/
#### Rapport 1: 

### Preprocessing Steps
#### Average Color - Steffen
#### small grayscale images  - Emil
#### Resize / Crop - Frederik

### Model
#### Database / other way to handle information - Roberta + Frederik
#### KNN model / evaluation - Steffen?
#### Kaggle submission format
#### How to modify KNN to do multilabel classification - Frederik

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


### Todo
#### ~~Resize and constant into downloader~~
#### How does multilabel KNN in sk-learn work? - Emil
#### ~~Feature extraction (Greyscale. Steffen's histograms.)~~
#### Memory handling (h5py, npz)
#### Do grid-search to determine K-NN. - Frederik
#### Write report. - Steffen, Frederik (data)
#### ~~Kaggle submission format into framework~~
#### ~~Graph for performance on each label~~
#### Looking into if the "missing" image has different labels attached to it. If so, remove them from traning set. - Frederik
#### Flash-talk points
