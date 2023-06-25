import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ImageProcessor import ImageProcessor
from Model import Model

# Set paths to data and csv files
train_data = os.path.join("data", "train")
train_csv = pd.read_csv(os.path.join("data", "augmented_train.csv"), index_col=0)
test_data = os.path.join("data", "test")
test_csv = pd.read_csv(os.path.join("data", "test.csv"), index_col=0)

# Data preprocessing
processor = ImageProcessor()
# train and validation data preprocessing
X_train, X_val, y_train, y_val = processor.create_dataset(train_csv, train_split=True)
# test data preprocessing
test_images, test_labels = processor.create_dataset(test_csv, train_split=False)
num_classes = len(set(y_train))


# Create model
model = Model(model_name="model_3")
model.create_model(num_classes=num_classes, image_shape=(32, 32, 3))
model.train_model(X_train, y_train, epochs=10, batch_size=32, X_val=X_val, y_val=y_val)
model.save_model(model_path="models")


# Evaluate Model
model = Model(model_name="model_3_test")
model.load_model("models/model_2.h5")

model.evaluate(test_images, test_labels)
model.save_model(model_path="models")
