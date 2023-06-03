import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    def augment_data(self) -> None:
        pass

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
