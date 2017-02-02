import numpy as np

def preprocess_input(images):
        return np.transpose(images, (0, 3, 1, 2))
