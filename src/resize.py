import os
import glob
import cv2

resize_dims = (256,256)

filenames = glob.glob('../data/raw_images/train/*.jpg') # not sorted

images = []
for img in filenames: # should use with?
    image = cv2.imread(img)
    image_resized = cv2.resize(image, resize_dims) # uses linear interpolation if enlarging
    
    new_dir = '../data/resized_images/train/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    cv2.imwrite(new_dir+img.split('/train/')[1][:-4]+'_rs.jpg', image_resized)