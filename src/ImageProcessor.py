import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ImageProcessor:
    def __init__(self) -> None:
        print("ImageProcessor initialized")

    def preprocess_images(
        self,
        image_path: str,
        image_size: tuple,
        convert_to_grayscale: bool = False,
        sharpen: bool = True,
    ) -> np.array:
        """Preprocess images for training and prediction. Apply some filters to the images.

        Args:
            image_path (str): path to image
            image_size (tuple): target image size
            convert_to_grayscale (bool, optional): Decide if image should be in grayscale or not. Defaults to True.
            sharpen (bool, optional): Decide if image should be sharpend. Defaults to True.

        Returns:
            np.array: image as numpy array
        """
        # read in and resize image
        image = cv.imread(image_path)
        image = cv.resize(image, image_size)

        # convert to grayscale and apply sharpen filter to image
        if convert_to_grayscale == True:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if sharpen == True:
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv.filter2D(image, -1, sharpen_kernel)

        # normalize pixel values
        image = image / 255.0
        return image

    def create_dataset(
        self, df: pd.DataFrame, train_split: bool, data_set_type: str
    ) -> tuple:
        """Create dataset for training and testing.

        Args:
            df (pd.DataFrame): DataFrame with path to images and corresponding labels
            train_split (bool): Decide if train data should be splitted into train and validation data
            data_set_type (str): Specify if train or test data should be processed

        Returns:
            tuple: tuple of numpy arrays with images and labels
        """
        images = list()
        labels = list()

        for i, row in tqdm(df.iterrows()):
            path = os.path.join(row["Path"])
            label = row["classId"]
            image = self.preprocess_images(
                image_path=path,
                image_size=(32, 32),
                convert_to_grayscale=False,
                sharpen=True,
            )
            if data_set_type == "test":
                image = np.expand_dims(image, axis=0)
            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        if train_split == True:
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=42
            )
            return X_train, X_val, y_train, y_val
        else:
            return images, labels

    def augment_image(
        self,
        image_path: str,
        rotation: bool = True,
        zoom: bool = True,
        noise: bool = True,
        rotation_range: tuple = (0, 90),
        zoom_range: tuple = (0.8, 1.2),
        noise_range: tuple = (0, 5),
    ) -> pd.DataFrame:
        """Augment images for training. Apply some filters to the images.

        Args:
            image_path (str): path to image
            rotation (bool, optional): Descide whether image should be rotated. Defaults to True.
            zoom (bool, optional): Descide wether zoom filter should be applied. Defaults to True.
            noise (bool, optional): Descide wether noise filter should be applied. Defaults to True.
            rotation_range (tuple, optional): Range in which the rotation factor is selected randomly . Defaults to (0, 90).
            zoom_range (tuple, optional): Range in which the zoom factor is selected randomly. Defaults to (0.8, 1.2).
            noise_range (tuple, optional): Range in which the noise factor is selected randomly. Defaults to (0, 5).

        Returns:
            pd.DataFrame: DataFrame with path to augmented image and corresponding classId
        """
        df_augmented = pd.DataFrame()

        image = cv.imread(image_path)

        if rotation == True:
            # create rotation matrix
            rotation_factor = np.random.uniform(*rotation_range)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv.getRotationMatrix2D(center, rotation_factor, 1.0)
            # rotate image
            rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
            # save image and add to dataframe
            rotated_image_path = os.path.join(
                f"{image_path.replace('.png' , '_rotated.png')}"
            )
            new_row = pd.DataFrame(
                {
                    "Width": 0,
                    "Height": 0,
                    "Roi.X1": 0,
                    "Roi.Y1": 0,
                    "Roi.X2": 0,
                    "Roi.Y2": 0,
                    "classId": int(image_path.split("/")[-2]),
                    "Path": rotated_image_path,
                },
                index=[0],
            )
            df_augmented = pd.concat([df_augmented, new_row]).reset_index(drop=True)
            cv.imwrite(rotated_image_path, rotated_image)

        if zoom == True:
            # apply zoom
            zoom_factor = np.random.uniform(*zoom_range)
            zoomed_image = cv.resize(image, None, fx=zoom_factor, fy=zoom_factor)
            # save image and add to dataframe
            zoomed_image_path = os.path.join(
                f"{image_path.replace('.png' , '_zoomed.png')}"
            )
            new_row = pd.DataFrame(
                {
                    "Width": 0,
                    "Height": 0,
                    "Roi.X1": 0,
                    "Roi.Y1": 0,
                    "Roi.X2": 0,
                    "Roi.Y2": 0,
                    "classId": int(image_path.split("/")[-2]),
                    "Path": zoomed_image_path,
                },
                index=[0],
            )
            df_augmented = pd.concat([df_augmented, new_row]).reset_index(drop=True)
            cv.imwrite(zoomed_image_path, zoomed_image)

        if noise == True:
            # apply noise to image
            noise_level = np.random.uniform(*noise_range)
            noise = np.random.normal(scale=noise_level, size=image.shape).astype(
                np.uint8
            )
            noisy_image = cv.add(image, noise)
            # save image and add to dataframe
            noisy_image_path = os.path.join(
                f"{image_path.replace('.png' , '_noisy.png')}"
            )
            new_row = pd.DataFrame(
                {
                    "Width": 0,
                    "Height": 0,
                    "Roi.X1": 0,
                    "Roi.Y1": 0,
                    "Roi.X2": 0,
                    "Roi.Y2": 0,
                    "classId": int(image_path.split("/")[-2]),
                    "Path": noisy_image_path,
                },
                index=[0],
            )
            df_augmented = pd.concat([df_augmented, new_row]).reset_index(drop=True)
            cv.imwrite(noisy_image_path, noisy_image)

        return df_augmented

    def show_image(self, image_path: str, predicted_label: int = 0) -> None:
        """Plot image with acutal label and predicted label.

        Args:
            image_path (str): path to image
            label (int): acutal label ot the image
            predicted_label (int, optional): predicted label. Defaults to 0.
        """
        image = cv.imread(image_path)

        meta = pd.read_csv("data/meta.csv")
        predicted_name = meta[meta["classId"] == predicted_label]["description"].values[
            0
        ]
        plot_title = f"Predicted label: {predicted_label} - {predicted_name}"

        # plot image with label and predicted label
        plt.imshow(image)
        plt.title(plot_title)
        plt.axis("off")
        plt.show()
