# LSDA
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
#### Repport 0: https://www.overleaf.com/15811080jqgcpbxscxdj#/60201263/
#### Repport 1: 

### Preprocessing Steps
#### Average Color - Steffen
#### small grayscale images  - Emil
#### Resize / Crop - Frederik

### Model
#### Database / other way to handle information - Roberta
#### KNN model / evaluation - Steffen?
#### Kaggle submission format

### Azue
login:
`lsda`
`CompetitionGroup4`

Connect with 
`ssh lsda@40.114.107.120`

Source into tensorflow enviroment:
`source lsdaenv/bin/activate`

# !!! DO NOT FORGET TO TURN OFF THE SERVER !!!
