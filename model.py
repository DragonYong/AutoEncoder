#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-16:25
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : model.py
# @Project  : 00PythonProjects
import tensorflow as tf
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
from tensorflow import keras

image_size = 28 * 28
h_dim = 20
z_dim = 20


class AE(tf.keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # 784 => 512
        self.fc1 = keras.layers.Dense(512)
        # 512 => h
        self.fc2 = keras.layers.Dense(h_dim)

        # h => 512
        self.fc3 = keras.layers.Dense(512)
        # 512 => image
        self.fc4 = keras.layers.Dense(image_size)

    def encode(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = (self.fc2(x))
        return x

    def decode_logits(self, h):
        x = tf.nn.relu(self.fc3(h))
        x = self.fc4(x)

        return x

    def decode(self, h):
        return tf.nn.sigmoid(self.decode_logits(h))

    def call(self, inputs, training=None, mask=None):
        # encoder
        h = self.encode(inputs)
        # decode
        x_reconstructed_logits = self.decode_logits(h)

        return x_reconstructed_logits


class VAE(tf.keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        # input => h
        self.fc1 = keras.layers.Dense(h_dim)
        # h => mu and variance
        self.fc2 = keras.layers.Dense(z_dim)
        self.fc3 = keras.layers.Dense(z_dim)

        # sampled z => h
        self.fc4 = keras.layers.Dense(h_dim)
        # h => image
        self.fc5 = keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        # mu, log_variance
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        """
        reparametrize trick
        :param mu:
        :param log_var:
        :return:
        """
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)

        return mu + eps * std

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        # encoder
        mu, log_var = self.encode(inputs)
        # sample
        z = self.reparameterize(mu, log_var)
        # decode
        x_reconstructed_logits = self.decode_logits(z)

        return x_reconstructed_logits, mu, log_var
