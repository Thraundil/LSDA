# Code from https://www.kaggle.com/nlecoy/imaterialist-downloader-util

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import json
import urllib3
import multiprocessing
import zipfile

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

from constants import *

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(fnames_and_urls):
    """
    download image, preprocess it and save it with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image = image.resize(IMAGES_DOWNLOAD_MIN_SIZE, Image.LANCZOS) # makes square images
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)

def parse_dataset(_dataset, _outdir, _max=10000000):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(_dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:_max]

def scrape(dataset, outdir, amount=None):
    """
    Downloads the files. 
    @param dataset: path to the dataset to scrape
    @param outdir: directory to save the files in
    """
    # reporting
    if amount is not None:
        print('Downloading %i files from %r' % (amount, dataset))
    else:
        print('Downloading all files from %r' % dataset)        
    print('Saving files in %r' % outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # parse json dataset file
    fnames_urls = parse_dataset(dataset, outdir)

    image_files = os.listdir(outdir)
    if len(image_files):
        last_id = sorted([int(image_file.split('.')[0]) for image_file in image_files])[-1]
        fnames_urls = fnames_urls[last_id:]

    # shorten the amount
    if amount is not None:
        fnames_urls = fnames_urls[:amount]

    # download data
    pool = multiprocessing.Pool(processes=50)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    sys.exit(1)

def unzip(fname, datadir):
    """
    If the given filename does not exist, unzips a file called "<fname>.zip"
    """
    if not os.path.exists(fname):
        with zipfile.ZipFile(fname + '.zip',"r") as zip_ref:
            zip_ref.extractall(datadir)
        

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # get args and create output directory
        dataset, outdir = sys.argv[1:]
        scrape(dataset, outdir)
    if len(sys.argv) == 4:
        # get args and create output directory
        dataset, outdir, amount = sys.argv[1:]
        scrape(dataset, outdir, amount=int(amount))
    elif len(sys.argv) == 1:
        print("No directories given, using defaults and downloading everything")

        labels = LABEL_DIR + '/train.json'
        outdir = RAW_IMAGES_DIR + '/train'
        datadir = LABEL_DIR
        unzip(labels, datadir)
        scrape(labels, outdir)

        labels = LABEL_DIR + '/validation.json'
        outdir = RAW_IMAGES_DIR + '/validation'
        unzip(labels, datadir)
        scrape(labels, outdir)

        labels = LABEL_DIR + '/test.json'
        outdir = RAW_IMAGES_DIR + '/test'
        unzip(labels, datadir)
        scrape(labels, outdir)
    else:
        print("error: not enough arguments")
        print("usage: python scraper.py [dataset] [outdir] [amount]")
        sys.exit(0)
