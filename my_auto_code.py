# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:07:32 2019

@author: 25493
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 100
INPUT_DIM = 784
HIDDEN_LAYER1 = 256
HIDDEN_LAYER2 = 64
LEARNING_RATE = 0.001
STEPS = 20000

mnist = input_data.read_data_sets("./", one_hot=True)

x = tf.placeholder("float", [None, INPUT_DIM])

def encoder(X):
    x1 = tf.layers.dense(X, units=HIDDEN_LAYER1)
    b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER1]))
    x1 = tf.add(x1, b1)
    sig1 = tf.nn.sigmoid(x1)
    x2 = tf.layers.dense(sig1, units=HIDDEN_LAYER2)
    b2 = tf.Variable(tf.random_normal([HIDDEN_LAYER2]))
    x2 = tf.add(x2, b2)
    sig2 = tf.nn.sigmoid(x2)
    return sig2

def decoder(X):
    x1 = tf.layers.dense(X, units=HIDDEN_LAYER1)
    b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER1]))
    x1 = tf.add(x1, b1)
    sig1 = tf.nn.sigmoid(x1)
    x2 = tf.layers.dense(sig1, units=INPUT_DIM)
    b2 = tf.Variable(tf.random_normal([INPUT_DIM]))
    x2 = tf.add(x2, b2)
    sig2 = tf.nn.sigmoid(x2)
    return sig2   

en_x = encoder(x)
de_x = decoder(en_x)

y_coder = de_x
y_true = x

loss = tf.reduce_mean(tf.pow(y_coder - y_true,2))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
optimizer = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEPS):
        train_data, _ = mnist.train.next_batch(BATCH_SIZE)
        loss_value, _ = sess.run([loss, optimizer],
                 feed_dict={x:train_data})
        if i%100==0:
            print("loss value is: ", loss_value)
    
    raw = []
    code = []
    for i in range(10):
        pic, _ = mnist.validation.next_batch(1)
        pic_x = sess.run(de_x, feed_dict={x:pic})
        code_pic = np.reshape(pic_x, [28, 28])
        code.append(code_pic)
        raw_pic = np.reshape(pic, [28, 28])
        raw.append(raw_pic)
    for i in range(10):
        plt.figure()
        plt.imshow(raw[i])
        plt.figure()
        plt.imshow(code[i])
    plt.show()