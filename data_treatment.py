import cv2
import numpy as np
import os

def img_cropping(image, filename):
    h, w = image.shape[:2]
    crop_h, crop_w = min(3000, h), min(3000, w)
    start_y = (h-crop_h) // 2
    start_x = (w-crop_w) // 2
    cropped_img = image[start_y:start_y+crop_h,start_x:start_x+crop_w]
    return cropped_img, f"{filename}_c"

# Resizing function, to reduce the number of pixel in the images.
def img_resizing(image, filename):
    resized_img = cv2.resize(image,(100,100))
    rows, cols = resized_img.shape[:2]
    return resized_img, rows, cols, f"{filename}_s"

def rotate_image(image, angle, rows, cols, filename):
    M = cv2.getRotationMatrix2D((cols / 2 , rows / 2 ), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image, f"{filename}_r{angle}"

def img_augmentation(image, filename):
    high_contrast = cv2.convertScaleAbs(image, alpha=1.5)
    low_contrast = cv2.convertScaleAbs(image, alpha=0.5)
    high_brightness = cv2.convertScaleAbs(image, beta=60)
    low_brightness = cv2.convertScaleAbs(image, beta=-60)
    return [
        (high_contrast, f"{filename}_hc"),
        (low_contrast, f"{filename}_lc"),
        (high_brightness, f"{filename}_hb"),
        (low_brightness, f"{filename}_lb")
    ]

def img_gaussian(image, std, filename):
    noise = np.random.normal(0, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image, f"{filename}_{std}"

def main(input_dir, output_dir):
    final_img = []
    input_img = []
    filenames = []

    for image in os.listdir(input_dir):
        file_path = os.path.join(input_dir, image)
        img = cv2.imread(file_path)
        if img is not None:
            input_img.append(img)
            filenames.append(os.path.splitext(image)[0])
        else:
            print("Failed to load image")

    cropped_img = []
    for image, filename in zip(input_img, filenames):
        cropped, cropped_filename = img_cropping(image, filename)
        cropped_img.append((cropped, cropped_filename))

    rotated_img = []
    angles = [0, 90, 180, 270]
    for image, filename in cropped_img:
        resized_img, rows, cols, new_filename = img_resizing(image, filename)
        for angle in angles:
            rotated_image, rotated_filename = rotate_image(resized_img, angle, rows, cols, new_filename)
            rotated_img.append((rotated_image, rotated_filename))
            final_img.append((rotated_image, rotated_filename))
    
    print(len(final_img))

    augmented_img = []
    for image, filename in rotated_img:
        (high_contrast_img, high_contrast_filename), (low_contrast_img, low_contrast_filename), (high_bright_img, high_bright_filename), (low_bright_img, low_bright_filename) = img_augmentation(image, filename)
        augmented_img.append((high_contrast_img, high_contrast_filename))
        augmented_img.append((low_contrast_img, low_contrast_filename))
        augmented_img.append((high_bright_img, high_bright_filename))    
        augmented_img.append((low_bright_img, low_bright_filename))
        final_img.append((high_contrast_img, high_contrast_filename))        
        final_img.append((low_contrast_img, low_contrast_filename))
        final_img.append((high_bright_img, high_bright_filename))    
        final_img.append((low_bright_img, low_bright_filename))
    
    print(len(final_img))

    standard_deviation = [25,75]
    for image, filename in augmented_img:
        for std in standard_deviation:
            gaussian_img, gaussian_filename = img_gaussian(image, std, filename)
            final_img.append((gaussian_img, gaussian_filename))

    print(len(final_img))

    
    os.makedirs(output_dir, exist_ok=True)
    saved_filenames = set()

    for image, filename in final_img:
        filepath = os.path.join(output_dir, f"{filename}.png")
        if filename not in saved_filenames:
            success = cv2.imwrite(filepath, image)
            if success:
                saved_filenames.add(filename)
            else:
                print(f"Failed to save image: {filepath}")
        else:
            print(f"Duplicate filename skipped : {filename}")
    return final_img

output_dir = r"C:\Users\camil\Documents\DSTI\Deep_Learning_with_Python\hand_gestures_categorization\output_images\\"
img_dir = r"C:\Users\camil\Documents\DSTI\Deep_Learning_with_Python\hand_gestures_categorization\input_images\\"

main(img_dir, output_dir)