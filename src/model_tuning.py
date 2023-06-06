import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import plot_model
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

from ImageProcessor import ImageProcessor


# create custom random search to print test accuracy after each epoch
class CustomReandomSearch(RandomSearch):
    def on_epoch_end(self, trial, model, epoch, logs):
        if epoch % 10 == 0:
            test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
            print(f"Test accuracy: {test_acc}")
        super().on_epoch_end(trial, model, epoch, logs)


def tune_model(params) -> keras.model:
    """Define model architecture and compile it

    Returns:
        keras.model: compiled model
    """
    model = keras.Sequential()
    model.add(
        Conv2D(
            params.Int("conv_1_filter", min_value=32, max_value=128, step=16),
            (3, 3),
            activation="relu",
            input_shape=(32, 32, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(
        Dense(
            params.Int("dense_1_units", min_value=64, max_value=256, step=32),
            activation="relu",
        )
    )
    model.add(Dense(43, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            params.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Prepare data
# Set paths to data and csv files
train_data = os.path.join("data", "train")
train_csv = pd.read_csv(os.path.join("data", "train.csv"), index_col=0)
test_data = os.path.join("data", "test")
test_csv = pd.read_csv(os.path.join("data", "test.csv"), index_col=0)

# Data preprocessing
processor = ImageProcessor()

# train and validation data preprocessing
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

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# test data preprocessing
test_images = list()
test_labels = list()
for i, row in tqdm(test_csv.iterrows()):
    path = os.path.join(row["Path"])
    label = row["classId"]
    image = processor.preprocess_images(
        image_path=path, image_size=(32, 32), convert_to_grayscale=False, sharpen=True
    )
    test_images.append(image)
    test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


# perform hyperparameter tuning
tuner = CustomReandomSearch(
    tune_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="tuned_models",
    project_name="traffic_signs",
)
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# get best hyperparameters and build model
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
for param in tuner.get_best_hyperparameters(num_trials=1).values():
    print(param)
best_model = tuner.hypermodel.build(best_hyperparameters)
best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


best_model.evaluate(test_images, test_labels)

best_model.save("models/model_3_tuned.h5")


# create different model architecture to compare
def create_model():
    model = keras.Sequential(
        [
            Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(43, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# train model
model = create_model()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.evaluate(test_images, test_labels)
# model.save("models/model_2.h5")
# model = keras.models.load_model("models/model_2.h5")

# plot model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# get model summary and plot loss functions
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(train_loss) + 1)

fig, axes = plt.subplots(figsize=(15, 5))

axes.plot(epochs, train_loss, "b", label="Training loss")
axes.plot(epochs, val_loss, "r", label="Validation loss")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.legend()
axes.set_title("Training and validation loss")
plt.savefig("models/loss.png")