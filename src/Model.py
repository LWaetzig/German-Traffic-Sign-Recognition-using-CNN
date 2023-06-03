import os
import cv2 as vs
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


class Model:
    def __init__(self, model_name: str = "model") -> None:
        self.model_name = model_name
        self.model = None
        self.accuracy = None
        print(f"Model: {self.model_name} created")

    def create_model(self, num_classes: int, image_shape: tuple = (32, 32, 3)):
        model = keras.Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=image_shape),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model = model

    def train_model(
        self, train_data, train_labels, epochs: int = 10, batch_size: int = 32, **kwargs
    ):
        model = self.model
        if model is None:
            print("No model to train")
            return
        if kwargs:
            print("Training with validation data")
            X_val = kwargs.get("X_val")
            y_val = kwargs.get("y_val")
            model.fit(
                train_data,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
            )
        else:
            model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    def predict(self, image: np.array) -> int:
        model = self.model
        if model is None:
            print("No model to predict")
            return

        prediction = model.predict(image)
        return prediction

    def evaluate(
        self,
        test_data: np.array,
        test_labels: np.array,
    ) -> None:
        pass

    def save_model(self, model_path: str = "models") -> None:
        """Save model to path. Model will be saved as .h5 file.

        Args:
            model_path (str, optional): Path to where model should be saved. Defaults to "models".
        """
        if self.model is None:
            print("No model to save")
            return

        if not os.path.exists(model_path):
            print(f"Created directory: {model_path}")
            os.mkdir(model_path)

        model_path = os.path.join(model_path, f"{self.model_name}.h5")
        self.model.save(model_path)
        print(f"Model {self.model_name} saved to {model_path}")

    def load_model(self, model_path: str = "model") -> None:
        """Load existing model from path. Model has to be saved as .h5 file.

        Args:
            model_path (str, optional): Path to model. Defaults to "model".
        """
        if not os.path.exists(model_path):
            print(f"No such file or directory to load model: {model_path}")
            return

        self.model = keras.models.load_model(model_path)
        print(f"Model {self.model_name} loaded from {model_path}")
