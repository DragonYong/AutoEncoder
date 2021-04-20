#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-16:49
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : VaritionalAutoEncoders.py
# @Project  : 00PythonProjects
import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras

from data import dateset_fashion_mnist
from model import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--DATA', help='inner batch size', default=None, type=str)
parser.add_argument('--IMAGE', help='inner batch size', default=None, type=str)
parser.add_argument('--BATCH_SIZE', help='inner batch size', default=100, type=int)
parser.add_argument('--EPOCHS', help='inner batch size', default=2, type=int)
parser.add_argument('--LEARNING_RATE', help='inner batch size', default=1e-3, type=float)
parser.add_argument('--IS_DRAW', help='weather to show graph', action='store_true', default=False)
args = parser.parse_args()

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# image grid
new_im = Image.new('L', (280, 280))

image_size = 28 * 28
h_dim = 512
z_dim = 20

model = VAE()
model.build(input_shape=(4, image_size))
model.summary()
optimizer = keras.optimizers.Adam(args.LEARNING_RATE)

dataset, num_batches = dateset_fashion_mnist(args)

for epoch in range(args.EPOCHS):

    for step, x in enumerate(dataset):

        x = tf.reshape(x, [-1, image_size])

        with tf.GradientTape() as tape:

            # Forward pass
            x_reconstruction_logits, mu, log_var = model(x)

            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            # Scaled by `image_size` for each individual pixel.
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstruction_logits)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss) / args.BATCH_SIZE
            # please refer to
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = - 0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
            kl_div = tf.reduce_mean(kl_div)

            # Backprop and optimize
            loss = tf.reduce_mean(reconstruction_loss) + kl_div

        gradients = tape.gradient(loss, model.trainable_variables)
        for g in gradients:
            tf.clip_by_norm(g, 15)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (step + 1) % 50 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, args.EPOCHS, step + 1, num_batches, float(reconstruction_loss), float(kl_div)))

    # Generative model
    z = tf.random.normal((args.BATCH_SIZE, z_dim))
    out = model.decode(z)  # decode with sigmoid
    out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
    out = out.astype(np.uint8)

    # since we can not find image_grid function from vesion 2.0
    # we do it by hand.
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = out[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    if not os.path.exists(args.IMAGE):
        os.mkdir(args.IMAGE)
    new_im.save(args.IMAGE + '/vae_sampled_epoch_%d.png' % (epoch + 1))
    if args.IS_DRAW:
        plt.imshow(np.asarray(new_im))
        plt.show()

    # Save the reconstructed images of last batch
    out_logits, _, _ = model(x[:args.BATCH_SIZE // 2])
    out = tf.nn.sigmoid(out_logits)  # out is just the logits, use sigmoid
    out = tf.reshape(out, [-1, 28, 28]).numpy() * 255

    x = tf.reshape(x[:args.BATCH_SIZE // 2], [-1, 28, 28])

    x_concat = tf.concat([x, out], axis=0).numpy() * 255.
    x_concat = x_concat.astype(np.uint8)

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = x_concat[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(args.IMAGE + '/vae_reconstructed_epoch_%d.png' % (epoch + 1))
    if args.IS_DRAW:
        plt.imshow(np.asarray(new_im))
        plt.show()
    print('New images saved !')
