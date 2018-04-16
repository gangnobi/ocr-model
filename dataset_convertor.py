#!/usr/bin/env python
"""This script are used to convert NIST Special Database 19 dataset to Tensorflow form to retrain
        - Black background
        - White character
        - JPG file
"""

import os
from os.path import join
import cv2

DATASET_PATH = join(os.getenv("HOME"), "Documents/NIST_Special_Database_19")
DATASET_ORIGINAL_PATH = join(DATASET_PATH, "by_class")
DATASET_RESULT_PATH = join(DATASET_PATH, "tensorflow_form")

if not os.path.exists(DATASET_RESULT_PATH):
    os.makedirs(DATASET_RESULT_PATH)

parentList = [f for f in os.listdir(
    DATASET_ORIGINAL_PATH) if os.path.isdir(join(DATASET_ORIGINAL_PATH, f))]

num = 0
num_file = []
name_file = {}

for p in parentList:
    print("reading with " + p + " ("+str(num)+"/62)")
    num += 1
    mypath = join(DATASET_ORIGINAL_PATH, p, "train_"+p)
    onlyfiles = [f for f in os.listdir(
        mypath) if os.path.isfile(join(mypath, f))]
    num_file.append(len(onlyfiles))
    name_file[p] = onlyfiles

num = 0
for p in parentList:
    print("working with " + p + " ("+str(num)+"/62)")
    num += 1
    newpath = join(DATASET_RESULT_PATH, p)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in range(min(num_file)):
        mypath = join(DATASET_ORIGINAL_PATH, p, "train_"+p)
        img = cv2.imread(join(mypath, name_file[p][i]))
        img = 255-img
        cv2.imwrite(join(newpath, name_file[p][i][:-3]) + 'jpg', img)
