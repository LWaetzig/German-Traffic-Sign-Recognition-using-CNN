import os

import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from tensorflow import keras
from tqdm import tqdm


class Model():
    def __init__(self, model_name: str = "model") -> None:
        self.model_name = model_name
        self.model = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        print(f"Model: {self.model_name} created")

    def create_model(self, num_classes: int, image_shape: tuple = (32, 32, 3)):
        model = keras.Sequential(
        [
            Conv2D(64, (3, 3), activation="relu", input_shape=image_shape),
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
            Dense(num_classes, activation="softmax"),
        ]
        )
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
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
        model = self.model
        if model is None:
            print("No model to evaluate")
            return

        predicted_labels = list()
        for image in tqdm(test_data):
            prediction = np.argmax(model.predict(image, use_multiprocessing=True))
            predicted_labels.append(prediction)

        self.accuracy = accuracy_score(test_labels, predicted_labels)
        self.precision = precision_score(test_labels, predicted_labels, average="macro")
        self.recall = recall_score(test_labels, predicted_labels, average="macro")
        self.f1_score = f1_score(test_labels, predicted_labels, average="macro")
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        class_report = classification_report(test_labels, predicted_labels)

        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 score: {self.f1_score}")
        print()
        print(f"Confusion matrix: \n{conf_matrix}")

        print()
        print(f"Classification report: \n{class_report}")

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
        if self.accuracy is not None:
            metrics = {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            }
            with open(os.path.join(model_path.replace("h5", ".json"), "w")) as f:
                f.write(str(metrics))


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


