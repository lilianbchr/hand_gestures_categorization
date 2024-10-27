# Hand Gestures Categorization
Deep Learning group project for DSTI 2024

## What is this project about?

This project is an attempt at recognizing hand gestures using deep learning.

## What is in this repository?

- `data_pretreatment.py`: script used to process and augment the data
- `model_training.ipynb`: notebook used to train the final sequential convolutional neural network using pytorch
- `sequential_model.py`: first try of a sequential convolutional neural network using tensorflow
- `predictor.py`: script used to predict the gesture of a new image using the trained model
- `crop_resize.py`: tool used to crop and resize the images selected by the user with predictor.py
- `requirements.txt`: file listing the dependencies of the project
- `models/`: folder containing the trained models (E10 for epoch 10, E20 for epoch 20)

## How to use this repository

- Run `pip install -r requirements.txt` to install the dependencies

If you want to retrain the model : 
- Run `python data_pretreatment.py` to crop, resize and augment the images.
- Run `model_training.ipynb` in google colab to train the model and save it in the `models` folder

If you want to use the model to predict the gesture of a new image : 
- Run `python predictor.py` to predict the gesture of a new image

## Authors
- Lilian Boucher 
- Quentin Camilleri
- Rabia Benabed
