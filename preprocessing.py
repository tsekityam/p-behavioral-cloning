import numpy as np

def get_cropped_images(images):
    shape = images[0].shape
    height = shape[0]
    width = shape[1]
    print(images.shape)
    return images[0:len(images), 40:height-20, 0:width]

def preprocess_input(images):
    cropped_images = get_cropped_images(images)
    return np.transpose(cropped_images, (0, 3, 1, 2))
