import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


def load_images_and_labels(image_dir):
    image_data = []
    labels = []

    for image in os.listdir(image_dir):
        file_path = os.path.join(image_dir, image)
        if image.endswith('.png') or image.endswith('.jpg'):
            img = cv2.imread(file_path)
            if img is not None:
                image_data.append(img)
                label = image.split('_')[0].lower()
                labels.append(label)
            else:
                print(f"Failed to load image : {image}")
    return np.array(image_data), labels

## TOUT CE QUI EST EN DESSOUS C'EST DU CHAT GPT QUE J'AI PAS EU LE TEMPS DE VERIFIER. C'EST POUR DONNER LA TRAME. 
## JE NE SAIS PAS A QUOI SERT LE class_names ET SI CA VAUT LE COUP DE LE GARDER
## POUR LE CNN JE VAIS TE PARTAGER DIRECTEMENT SUR DRIVE MON NOTEBOOK QUE J'AI CREE, TU PEUX T'EN INSPIRER VOIR MEME COPIER COLLER
## ET VOIR LES RESULTATS QU'ON OBTIENT AVEC LE TRAINING, ET MODIFIER EN FONCTION, COMME J'AI FAIT

def one_hot_encode_labels(labels):
    lb = LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels)
    return one_hot_labels, lb.classes_

def prepare_images_and_labels_for_ml(image_dir):

    images, labels = load_images_and_labels(image_dir)

    one_hot_labels, class_names = one_hot_encode_labels(labels)
    
    return images, one_hot_labels, class_names

image_dir = r'C:\Users\camil\Documents\DSTI\Deep_Learning_with_Python\input_images'


images, one_hot_labels, class_names = prepare_images_and_labels_for_ml(image_dir)


X_train, X_val, y_train, y_val = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

# CNN model definition
def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the convolutional layers to feed into Dense layers
    model.add(Flatten())

    # Fully connected Dense layer
    model.add(Dense(128, activation='relu'))

    # Output layer for classification
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Create the model with input shape (height, width, channels) and number of classes
input_shape = images.shape[1:]  # e.g., (100, 100, 3) for 100x100 RGB images
num_classes = one_hot_labels.shape[1]  # Number of unique classes in the dataset

model = create_cnn_model(input_shape, num_classes)

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")