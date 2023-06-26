import os

import pandas as pd

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
X_train, X_val, y_train, y_val = processor.create_dataset(
    train_csv, train_split=True, data_set_type="train"
)
# test data preprocessing
test_images, test_labels = processor.create_dataset(
    test_csv, train_split=False, data_set_type="test"
)

num_classes = len(set(y_train))

# Create model
model = Model(model_name="model_3")
model.create_model(num_classes=num_classes, image_shape=(32, 32, 3))
model.train_model(X_train, y_train, epochs=10, batch_size=64, X_val=X_val, y_val=y_val)

# Evaluate Model
model.evaluate(test_images, test_labels)
model.save_model(model_path=os.path.join("models"))
