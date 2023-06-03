import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ImageProcessor import ImageProcessor
from Model import Model

# Set paths to data and csv files
train_data = os.path.join("data", "train")
train_csv = pd.read_csv(os.path.join("data", "train.csv"), index_col=0)
test_data = os.path.join("data", "test")
test_csv = pd.read_csv(os.path.join("data", "test.csv"), index_col=0)


# Data preprocessing
processor = ImageProcessor()

images = list()
labels = list()

for i, row in tqdm(train_csv.iterrows()):
    path = os.path.join(row["Path"])
    label = row["classId"]
    image = processor.preprocess_images(
        image_path=path, image_size=(32, 32), convert_to_grayscale=False, sharpen=True
    )
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Split data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
num_classes = len(set(y_train))

# Create model
model = Model(model_name="model_1")
model.create_model(num_classes=num_classes, image_shape=(32, 32, 3))
model.train_model(X_train, y_train, epochs=10, batch_size=32, X_val=X_val, y_val=y_val)
model.save_model(model_path="models")


# Test model
model = Model(model_name="model_1_test")
model.load_model(model_path="models/model_1.h5")
test_image = processor.preprocess_images(image_path="data/Test/00000.png", image_size=(32, 32), convert_to_grayscale=False, sharpen=True)
test_image = np.expand_dims(test_image, axis=0)
test_image = np.array(test_image)
prediction = np.argmax(model.predict(test_image))


# Plot image with prediction
processor.show_image(image_path="data/Test/00000.png", label=16, predicted_label=prediction)