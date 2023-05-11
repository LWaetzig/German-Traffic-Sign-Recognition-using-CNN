import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2 #function for image editing

#import the data
path = 'data/Train'
classes = os.listdir(path)
data = []
labels = []

#label the data
for i, c in enumerate(classes):
    class_path = os.path.join(path, c)
    for img_file in os.listdir(class_path):
        try:
            img = cv2.imread(os.path.join(class_path, img_file))
            img = cv2.resize(img, (30, 30))
            img_arr = np.array(img)
            data.append(img_arr)
            labels.append(i)
        except:
            print(f"Error {img_file}")






