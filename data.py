#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-17:00
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : data.py
# @Project  : 00PythonProjects
import gzip
import os

import tensorflow as tf
import numpy as np
from tensorflow import keras


def get_fashion_mnist(path):
    """
    files的顺序不能自定义，否则解压会出错
    """
    # files = os.listdir(path)
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for file in files:
        paths.append(os.path.join(path, file))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def dateset_fashion_mnist(args):
    # 对与离线的无法访问外网，修改源代码，可以加载本地数据
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = get_fashion_mnist(args.DATA)

    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

    # In[19]:

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # we do not need label
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(args.BATCH_SIZE * 5).batch(args.BATCH_SIZE)

    num_batches = x_train.shape[0] // args.BATCH_SIZE
    return dataset, num_batches


def dataset_mnist(args):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(args.DATA)
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # we do not need label
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(args.BATCH_SIZE * 5).batch(args.BATCH_SIZE)

    num_batches = x_train.shape[0] // args.BATCH_SIZE
    return dataset, num_batches


if __name__ == '__main__':
    path = "/media/turing/D741ADF8271B9526/DATA/tensorflow/keras/fashion-mnist"
    (x_train, y_train), (x_test, y_test) = get_fashion_mnist(path)
    print(x_test)
