import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_resize import crop_image


# Definition of the CNN model architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load the saved model
model = ConvNet(num_classes=10)
model.load_state_dict(torch.load("models/model_E10.pth", map_location=torch.device("cpu")))
model.eval()

# Create a file dialog to choose an image
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

if file_path:
    image = cv2.imread(file_path)

    if image is not None:
        print(f"Image loaded successfully: {file_path}")
        image = crop_image(image)
        image = cv2.resize(image, (100, 100))
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0) / 255.0

        # Make a prediction
        model.eval()
        with torch.no_grad():
            output = model(image)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        class_names = [
            "crossed",
            "fine",
            "finger",
            "halt",
            "little",
            "peace",
            "rock",
            "spock",
            "surfer",
            "thumb",
        ]
        predicted_label = class_names[predicted_class]

        print(f"Predicted Label: {predicted_label}")
    else:
        print(f"Failed to load image: {file_path}")
else:
    print("No file selected.")
