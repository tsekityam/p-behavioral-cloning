# python model.py --drive_log_file driving_data.csv

import csv
import cv2
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import matplotlib.image as mpimg
import numpy as np
import os
import preprocessing
import random
import scipy.ndimage as sndi
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('drive_log_file', '', "Drive log file (.csv)")

# model developed by comma.ai
def get_model(input_shape):
    ch = input_shape[0]
    row = input_shape[1]
    col = input_shape[2]

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
    input_shape=(ch, row, col),
    output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", W_regularizer=l2(0.0005), b_regularizer=l2(0.01)))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.0005), b_regularizer=l2(0.01)))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.0005), b_regularizer=l2(0.01)))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.8))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

# # model developed by Nvidia
# def get_model(input_shape):
#     model = Sequential()
#     model.add(Lambda(lambda x: x/127.5 - 1.,
#     input_shape=input_shape,
#     output_shape=input_shape))
#     model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu', border_mode='same', name='Conv1'))
#     model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu', border_mode='same', name='Conv2'))
#     model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu', border_mode='same', W_regularizer=l2(0.0005), b_regularizer=l2(0.0005), name='Conv3'))
#     model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', border_mode='same', W_regularizer=l2(0.0005), b_regularizer=l2(0.0005), name='Conv4'))
#     model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', border_mode='same', W_regularizer=l2(0.0005), b_regularizer=l2(0.0005), name='Conv5'))
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(1164, activation='elu', name='Dens1'))
#     model.add(Dropout(0.8))
#     model.add(Dense(100, activation='elu', name='Dens2'))
#     model.add(Dense(50, activation='elu', name='Dens3'))
#     model.add(Dense(10, activation='elu', name='Dens4'))
#     model.add(Dense(1))
#
#     model.compile(optimizer="adam", loss="mse")
#     model.optimizer.lr.assign(0.0001)
#
#     return model

n_samples = 500

def image_generator(X_image, X_flip, y_steering, batch_size):
    sample_image_shape = mpimg.imread(X_image[0]).shape

    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, sample_image_shape[0], sample_image_shape[1], sample_image_shape[2]))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            #choose random index in features
            index = np.random.randint(len(X_image))
            image = mpimg.imread(X_image[index])
            if X_flip[index] == 0:
                batch_features[i] = image
                batch_labels[i] = y_steering[index]
            else:
                batch_features[i] = cv2.flip(image, flipCode=1)
                batch_labels[i] = y_steering[index]

        yield preprocessing.preprocess_input(batch_features), batch_labels

def main(_):
    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    X_image = []
    X_flip = []
    y_steering = []

    batch_size = 512

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

    sample_image = preprocessing.preprocess_input(np.array([mpimg.imread(X_image[0])]))[0]
    print('Image Shape: ', sample_image.shape)

    X_train_image, X_test_image, X_train_flip, X_test_flip, y_train_steering, y_test_steering = train_test_split(X_image, X_flip, y_steering, test_size=0.2)

    print('Training Data Count:', len(X_train_image))
    print('Validation Data Count:', len(X_test_image))

    X_train_image, X_train_flip, y_train_steering = shuffle(X_train_image, X_train_flip, y_train_steering)

    model = get_model(sample_image.shape)
    model.fit_generator(image_generator(X_train_image, X_train_flip, y_train_steering, batch_size), samples_per_epoch=1024*12, nb_epoch=50, validation_data=image_generator(X_test_image, X_test_flip, y_test_steering, batch_size), nb_val_samples=1024*5, callbacks=[EarlyStopping(patience=1, verbose=1), ReduceLROnPlateau( patience=0, min_lr=0.00001, factor=0.5, verbose=1)])

    print("Saving model weights and configuration file.")

    # model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
