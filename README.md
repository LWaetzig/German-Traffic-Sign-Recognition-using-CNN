# Data-Exploration


## What is this?
TLDR: This is a German Traffic Sign Recognition based on a CNN.

Project for the Lecture Data Exploration at DHBW Mannheim. The goal is to detect German Traffic Signs with a CNN. The dataset ist from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and contains 43 different classes with 39.209 training images and 12.630 test images.

Contributors:
  - [Nicholas Link](https://github.com/Nicho-Link)
  - [Alexander Paul](https://github.com/alexx1374)
  - [Lucas Wätzig](https://github.com/LWaetzig)

## Directory Structure
- [data](data): contains the dataset
- [models](models): contains the trained models
  - model_1: vanilla model (without any tuning) -> accuracy: 82%
  - model_2: trained with 30 epochs and a batch size of 32 -> accuracy: 91%
  - model_3: trained with 10 epochs and a batch size of 64 -> accuracy: 97%
- [src](src): containts required files for classification
  - [augment_data.py](src/augment_data.py): script to augment the train dataset
  - [ImageProcessor.py](src/ImageProcessor.py): class to process the images
  - [model_creation.py](src/model_creation.py): script to create the model
  - [model_tuning.py](src/model_tuning.py): script to tune the model
  - [Model.py](src/Model.py): class to initiate the model
  
- [main.py](main.py): script to run the detection with a trained model

## Run Application
- install the required [packages](requirements.txt)
```bash
pip install -r requirements.txt
```
- make sure there is at least one **.h5** file in the models directory
- to run the classification with a trained model, run the following command in a shell:
```python
python main.py -i "path/to/image" -m "path/to/model"
```
- passable arguments:
  - -i / --image: path to the image
  - -m / --model: path to the model

- example:
```python
python main.py -i "data/Examples/test1.jpg" -m "models/model_3.h5"
```



## Object Detection
We also developed an object detection using a pretrained model from the tensorflow model zoo and the tensorflow api. Due to a huge amount of computing power we used Google Colab. Follow this [link](https://colab.research.google.com/drive/1KaH1XNzDOWr7oWnatquFtEvylNYIjjBk#scrollTo=n-74rTbe32Iw) to the notebook.
If you want to use this notebook, follow the instructions in the notebook. You have to import the necessary data -> 
- [Images](https://drive.google.com/file/d/1cWV0v5xCU8sR588Bv2Z_FRitkhB0Rhl0/view?usp=sharing)
- [Model](https://drive.google.com/drive/folders/1Z9528-ZzwHr-KPHkeT4BjLhKx9rwFjys?usp=sharing)

