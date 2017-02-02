import csv
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import matplotlib.image as mpimg
import numpy as np
import os
import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('drive_log_file', '', "Drive log file (.csv)")

batch_size = 256
batch_index = 0

def get_sample_image():
    with open(FLAGS.drive_log_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)

        next(reader)    # skip the first row

        row = next(reader)
        return preprocessing.preprocess_input([mpimg.imread(row[0])])[0]


def get_training_data_count():
    with open(FLAGS.drive_log_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        return len(list(reader)) - 1


# model developed by comma.ai
def get_model(input_shape):
    ch = input_shape[0]
    row = input_shape[1]
    col = input_shape[2]

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
    input_shape=(ch, row, col),
    output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

n_samples = 500

def image_generator():
    global batch_index
    X_center = []
    # X_left = []
    # X_right = []
    y_steering = []
    # y_throttle = []
    # y_brake = []
    # y_speed = []
    with open(FLAGS.drive_log_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)

        next(reader)    # skip the first row

        if batch_size * batch_index > get_training_data_count():
            batch_index = 0

        # skip data used in previous batch
        for i in range(batch_size * batch_index):
            next(reader)

        for i in range(batch_size):
            row = next(reader)
            X_center.append(mpimg.imread(row[0]))
            # X_left.append(mpimg.imread(row[1]))
            # X_right.append(mpimg.imread(row[2]))
            y_steering.append(row[3])
            # y_throttle.append(row[4])
            # y_brake.append(row[5])
            # y_speed.append(row[6])

    X_center = np.array(X_center)
    # X_left = np.array(X_left)
    # X_right = np.array(X_right)
    y_steering = np.array(y_steering)
    # y_throttle = np.array(y_throttle)
    # y_brake = np.array(y_brake)
    # y_speed = np.array(y_speed)

    X_train, y_train = preprocessing.preprocess_input(X_center), y_steering
    batch_index = batch_index + 1
    # yield X_train, y_train


    while True:
        yield X_train, y_train

def main(_):
    print('Image Shape: ', get_sample_image().shape)
    print('Training Data Count:', get_training_data_count())


    model = get_model(get_sample_image().shape)
    model.fit_generator(image_generator(), samples_per_epoch=get_training_data_count(), nb_epoch=10, nb_val_samples=get_training_data_count() * 0.2)

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
