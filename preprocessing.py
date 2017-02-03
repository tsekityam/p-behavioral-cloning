import numpy as np
import cv2

def get_cropped_images(images):
    shape = images[0].shape
    height = shape[0]
    width = shape[1]
    return images[0:len(images), 40:height-20, 0:width]

def get_images_in_hsv(images):
    hsv_images = []
    for i in range(len(images)):
        hsv_images.append(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
    return np.array(hsv_images)

def preprocess_input(images):
    cropped_images = get_cropped_images(images)
    hsv_cropped_images = get_images_in_hsv(cropped_images)
    return np.transpose(hsv_cropped_images, (0, 3, 1, 2))
