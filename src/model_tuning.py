import os

import keras
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras_tuner.tuners import RandomSearch
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

from ImageProcessor import ImageProcessor


def create_model():
    """Definde model architecture and compile it

    Returns:
        keras.model: compiled keras model
    """
    model = keras.Sequential(
        [
            keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(43, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def tune_model(params):
    """Define model architecture and compile it

    Returns:
        keras.model: compiled keras model
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
    model.add(
        Conv2D(
            params.Int("conv_2_filter", min_value=32, max_value=128, step=16),
            (3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            params.Int("conv_3_filter", min_value=64, max_value=256, step=32),
            (3, 3),
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            params.Int("conv_4_filter", min_value=64, max_value=128, step=32),
            (3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(
        Dense(
            params.Int("dense_1_units", min_value=64, max_value=512, step=64),
            activation="relu",
        )
    )
    model.add(
        Dropout(params.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1))
    )
    model.add(
        Dense(
            params.Int("dense_2_units", min_value=64, max_value=256, step=32),
            activation="relu",
        )
    )
    model.add(
        Dropout(params.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1))
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
train_csv = pd.read_csv(os.path.join("data", "augmented_train.csv"), index_col=0)
test_data = os.path.join("data", "test")
test_csv = pd.read_csv(os.path.join("data", "test.csv"), index_col=0)

# Data preprocessing
processor = ImageProcessor()

# train and validation data preprocessing
X_train, X_val, y_train, y_val = processor.create_dataset(train_csv, train_split=True, data_set_type="train")
# test data preprocessing
test_images, test_labels = processor.create_dataset(test_csv, train_split=False, data_set_type="test")

# perform Grid Search
keras_model = KerasClassifier(model=create_model, verbose=1)

param_grid = {
    "epochs": [10, 20, 30],
    "batch_size": [16, 32, 64],
}
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# perform hyperparameter tuning
tuner = RandomSearch(
    tune_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="tuned_models",
    project_name="traffic_signs",
)
tuner.search(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

# get best hyperparameters and build model
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
for param in tuner.get_best_hyperparameters(num_trials=1).values():
    print(param)
best_model = tuner.hypermodel.build(best_hyperparameters)
best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

best_model.evaluate(test_images, test_labels)

best_model.save("models/model_3_tuned.h5")
