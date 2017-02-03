import numpy as np
import cv2
import scipy.ndimage as sndi

def get_cropped_images(images):
    shape = images[0].shape
    height = shape[0]
    width = shape[1]
    return images[0:len(images), 65:height-25, 0:width]

def get_images_in_hsv(images):
    hsv_images = []
    for i in range(len(images)):
        hsv_images.append(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
    return np.array(hsv_images)

def get_resized_images(images):
    return sndi.zoom(images, [1, 1/3, 1/3, 1])

def preprocess_input(images):
    cropped_images = get_cropped_images(images)
    # resized_cropped_images = get_resized_images(cropped_images)
    # resized_cropped_images_hsv = get_images_in_hsv(resized_cropped_images)
    return np.transpose(cropped_images, (0, 3, 1, 2))
