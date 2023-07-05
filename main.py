import argparse
import os

import numpy as np

from src.ImageProcessor import ImageProcessor
from src.Model import Model


def main(args):
    model_path = os.path.join(args.model)
    image_path = os.path.join(args.image)

    # check if image and model exist
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist")
        return

    print("Start Classification")
    model = Model(model_name="Classifier")
    model.load_model(model_path=model_path)

    print("Preprocess image")
    processor = ImageProcessor()
    image = processor.preprocess_images(
        image_path=image_path,
        image_size=(32, 32),
        convert_to_grayscale=False,
        sharpen=True,
    )
    image = np.expand_dims(image, axis=0)
    image = np.array(image)

    print("Try to predict image")
    prediction = np.argmax(model.predict(image))

    processor.show_image(image_path=image_path, predicted_label=prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", "-i", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to model to use for prediction",
    )
    args = parser.parse_args()

    main(args)
