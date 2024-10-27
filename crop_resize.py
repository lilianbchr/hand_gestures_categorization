import os

import cv2


def crop_image(image):
    h, w = image.shape[:2]
    crop_size = min(3000, h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    cropped_img = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    return cropped_img


def main(input_dir, output_dir):
    final_images = []
    input_images = []
    image_names = []  # List to store image file names

    for image in os.listdir(input_dir):
        file_path = os.path.join(input_dir, image)
        img = cv2.imread(file_path)
        if img is not None:
            input_images.append(img)
            image_names.append(image)  # Store the file name
        else:
            print("Failed to load image")

    for image, name in zip(input_images, image_names):  # Use zip to iterate over images and names
        cropped_image = crop_image(image)
        resized_image = cv2.resize(cropped_image, (100, 100))
        final_images.append(resized_image)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, name)  # Use the stored file name
        cv2.imwrite(output_path, resized_image)
        print(f"Saved image: {output_path}")


if __name__ == "__main__":
    input_dir = (
        r"C:\Users\balle\Documents\DSTI\DL\hand_gestures_categorization\resources\test_image"
    )
    output_dir = (
        r"C:\Users\balle\Documents\DSTI\DL\hand_gestures_categorization\resources\test_image\output"
    )
    main(input_dir, output_dir)
