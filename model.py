# python model.py --drive_log_file driving_log.csv

import csv
import cv2
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2
import matplotlib.image as mpimg
import numpy as np
import os
import preprocessing
import random
import scipy.ndimage as sndi
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('drive_log_file', '', "Drive log file (.csv)")

# # model developed by comma.ai
# def get_model(input_shape):
#     ch = input_shape[0]
#     row = input_shape[1]
#     col = input_shape[2]
#
#     model = Sequential()
#     model.add(Lambda(lambda x: x/127.5 - 1.,
#     input_shape=(ch, row, col),
#     output_shape=(ch, row, col)))
#     model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
#     model.add(ELU())
#     model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
#     model.add(ELU())
#     model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
#     model.add(Flatten())
#     model.add(Dropout(.5))
#     model.add(ELU())
#     model.add(Dense(512))
#     model.add(Dropout(.8))
#     model.add(ELU())
#     model.add(Dense(1))
#
#     model.compile(optimizer="adam", loss="mse")
#
#     return model

# model developed by Nvidia
def get_model(input_shape):
    row = input_shape[0]
    col = input_shape[1]
    ch = input_shape[2]

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
    input_shape=(row, col, ch),
    output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu', border_mode="same", W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu', border_mode="same", W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu', border_mode="same", W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', border_mode="same", W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', border_mode="same", W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', W_regularizer=l2(0.01)))
    model.add(Dense(100, activation='elu', W_regularizer=l2(0.01)))
    model.add(Dense(50, activation='elu', W_regularizer=l2(0.01)))
    model.add(Dense(10, activation='elu', W_regularizer=l2(0.01)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

n_samples = 500

def image_generator(X_image, X_flip, y_steering, batch_size):
    sample_image_shape = mpimg.imread(X_image[0]).shape

    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, sample_image_shape[0], sample_image_shape[1], sample_image_shape[2]))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        batch_index = np.random.randint(len(X_image)/batch_size)
        for i in range(batch_size):
            #choose random index in features
            image = mpimg.imread(X_image[batch_index + i])
            if X_flip[batch_index + i] == 0:
                batch_features[i] = image
                batch_labels[i] = y_steering[batch_index + i]
            else:
                batch_features[i] = cv2.flip(image, flipCode=1)
                batch_labels[i] = y_steering[batch_index + i]

        yield preprocessing.preprocess_input(batch_features), batch_labels

def main(_):
    X_image = []
    X_flip = []
    y_steering = []

    batch_size = 256

    with open(FLAGS.drive_log_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)

        for row in reader:
            X_image.append(row[0])
            X_flip.append(float(row[1]))
            y_steering.append(float(row[2]))

    X_image = np.array(X_image)
    X_flip = np.array(X_flip)
    y_steering = np.array(y_steering)

    if len(X_image) == 0:
        print('No data found')
        return

    sample_image = mpimg.imread(X_image[0])
    print('Image Shape: ', sample_image.shape)
    print('Training Data Count:', len(X_image))


    model = get_model(preprocessing.preprocess_input(np.array([sample_image]))[0].shape)
    model.fit_generator(image_generator(X_image, X_flip, y_steering, batch_size), samples_per_epoch=len(X_image), nb_epoch=5)

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
