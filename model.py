import csv
import matplotlib.image as mpimg
import numpy as np
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


def main(_):
    # load training data
    X_train, y_train = load_data(FLAGS.drive_log_file)

    print('Image Shape: ', X_train.shape[1:])
    print('Training Data Count:', len(X_train))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
