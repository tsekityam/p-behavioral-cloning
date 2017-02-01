import csv
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import matplotlib.image as mpimg
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('drive_log_file', '', "Drive log file (.csv)")


def load_data(drive_log):
    X_center = []
    X_left = []
    X_right = []
    y_steering = []
    y_throttle = []
    y_brake = []
    y_speed = []
    with open(drive_log, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)

        next(reader)    # skip the first row

        # TODO: load all rows from given data
        for i in range(1000):
            row = next(reader)
            X_center.append(mpimg.imread(row[0]))
            X_left.append(mpimg.imread(row[1]))
            X_right.append(mpimg.imread(row[2]))
            y_steering.append(row[3])
            y_throttle.append(row[4])
            y_brake.append(row[5])
            y_speed.append(row[6])

    X_center = np.array(X_center)
    X_left = np.array(X_left)
    X_right = np.array(X_right)
    y_steering = np.array(y_steering)
    y_throttle = np.array(y_throttle)
    y_brake = np.array(y_brake)
    y_speed = np.array(y_speed)

    X_train, y_train = X_center, y_steering

    return X_train, y_train

# model developed by comma.ai
def get_model(input_shape):
    row = input_shape[0]
    col = input_shape[1]
    ch = input_shape[2]

    print(row, col, ch)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
    input_shape=(row, col, ch),
    output_shape=(row, col, ch)))
    model.add(Convolution2D(8, 8, 16, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(5, 5, 32, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(5, 5, 64, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def main(_):
    # load training data
    X_train, y_train = load_data(FLAGS.drive_log_file)

    print('Image Shape: ', X_train.shape[1:])
    print('Training Data Count:', len(X_train))

    model = get_model(X_train.shape[1:])
    model.fit(X_train, y_train, nb_epoch=10, validation_split=0.2)

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
