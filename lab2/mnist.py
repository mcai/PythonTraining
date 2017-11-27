import os
import gzip
from urllib.request import urlretrieve

import numpy as np

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'


# Download and import the MNIST dataset from Yann LeCun's website.
# Reserve 10,000 examples from the training set for validation.
# Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).
def load_data(one_hot=True, reshape=None, validation_size=10000):
    x_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    x_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    x_train = x_train[:-validation_size]
    y_train = y_train[:-validation_size]

    if one_hot:
        y_train, y_test = [to_one_hot(y) for y in (y_train, y_test)]

    if reshape:
        x_train, x_test = [x.reshape(*reshape) for x in (x_train, x_test)]

    return x_train, y_train, x_test, y_test


def load_images(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)


def load_labels(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Download the file, unless it's already here.
def maybe_download(filename):
    if not os.path.exists(filename):
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, filename)


# Convert class labels from scalars to one-hot vectors.
def to_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]
