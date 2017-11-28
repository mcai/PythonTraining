import os
import gzip
from urllib.request import urlretrieve

import numpy as np

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'


# 从Yann LeCun的网站下载和导入mnist数据.
# 从训练集中保留10,000个样本用作验证数据.
# 每幅图片是一个包含784（28x28）个元素的（浮点数，从0（白色）到1（黑色））数组.
def load_data(one_hot=True, validation_size=10000):  # 载入数据
    x_train = load_images('train-images-idx3-ubyte.gz')  # 载入训练集的图片
    y_train = load_labels('train-labels-idx1-ubyte.gz')  # 载入训练集的标签
    x_test = load_images('t10k-images-idx3-ubyte.gz')  # 载入测试集的图片
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')  # 载入测试集的标签

    x_train = x_train[:-validation_size]  # 从训练集的图片中除去验证数据
    y_train = y_train[:-validation_size]  # 从训练集的标签中除去验证数据

    if one_hot:
        y_train, y_test = [to_one_hot(y) for y in (y_train, y_test)]

    return x_train, y_train, x_test, y_test


def load_images(filename):  # 从指定的文件中载入图片
    download_if_necessary(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)


def load_labels(filename):  # 从指定的文件中载入标签
    download_if_necessary(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def download_if_necessary(filename):  # 检测指定的文件是否存在，若不存在则下载该文件
    if not os.path.exists(filename):
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, filename)


def to_one_hot(labels, num_classes=10):  # 将标签由标量转换为one-hot向量
    return np.eye(num_classes)[labels]
