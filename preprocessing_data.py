import os
import cv2

def load_images_and_labels(image_dir):
    image_paths = []
    labels = []

    for image in os.listdir(image_dir):
        file_path = os.path.join(image_dir, image)
        if image.endswith('.png') or image.endswith('.jpg'):
            img = cv2.imread(file_path)
            if img is not None:
                image_paths.append(img)
                label = image.split('_')[0].lower()
                labels.append(label)
            else:
                print(f"Failed to load image : {image}")
    return image_paths, labels

image_dir = r"C:\Users\camil\Documents\DSTI\Deep_Learning_with_Python\input_images\\"
images, label = load_images_and_labels(image_dir)
print(label[22:29])