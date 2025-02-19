import numpy as np
import cv2
import random


def augment_ground_image(image):

    height, width = image.shape[:2]

    # Random crop parameters
    crop_scale = random.uniform(0.7, 1.0)  # Crop between 70% to 100% of the image size
    crop_h = int(height * crop_scale)
    crop_w = int(width * crop_scale)
    x_start = random.randint(0, width - crop_w)
    y_start = random.randint(0, height - crop_h)
    cropped = image[y_start:y_start + crop_h, x_start:x_start + crop_w]

    # Resize back to original size
    resized = cv2.resize(cropped, (width, height))

    # Adjust brightness: alpha > 1 brightens, alpha < 1 darkens.
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)  # Adding some brightness offset
    bright_img = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

    return bright_img


num_ground_img = 100
save_path = 'temp/'

for i in range(num_ground_img):
    ground_img = cv2.imread("floor_1.png")
    augmented_img = augment_ground_image(ground_img)

    # Save the augmented image temporarily
    augmented_path = save_path + f"aug_floor_{i}.png"
    cv2.imwrite(augmented_path, augmented_img)

