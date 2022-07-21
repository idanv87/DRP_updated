import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants
from utils import *
from data_generator import create_test_data

def complete(H, kernelleft, kernelright, kernelup, kerneldown):
    rowup = tf.nn.conv2d(H, kernelup, strides=1, padding='VALID')
    rowdown = tf.nn.conv2d(H, kerneldown, strides=1, padding='VALID')
    a = tf.concat([rowup, H, rowdown], axis=1)
    colleft = tf.nn.conv2d(a, kernelleft, strides=1, padding='VALID')
    colright = tf.nn.conv2d(a, kernelright, strides=1, padding='VALID')

    return tf.concat([colleft, a, colright], axis=2)

n = 91
Hx = np.random.rand(1, n - 2, n - 1, 1)
Hy = np.random.rand(1, n - 1, n - 2, 1)

k1 = np.array([[35 / 16, -35 / 16, 21 / 16, -5 / 16]])
k2 = np.array([[4, -6, 4, -1]])

A = np.append(k1, np.zeros((1, n - 4)))
B = np.append(k2, np.zeros((1, n - 4)))
print(A.shape)

kernelleft = np.reshape(A, [1, A.shape[0], 1, 1])
assert kernelleft.shape == (1, n, 1, 1)
kernelright = tf.reverse(kernelleft, [1])

kernelup = np.reshape(B, [B.shape[0], 1, 1, 1])
assert kernelup.shape == (n, 1, 1, 1)
kerneldown = tf.reverse(kernelup, [0])





x = np.linspace(1, 2, n)
dx = 1 / (n - 1)
X, Y = np.meshgrid(x, x, indexing='ij')
Xa, Ya = np.meshgrid(x - dx, x + dx / 2, indexing='ij')
f = (np.cos(X) + np.exp(Y)).reshape(1, n, n, 1)
g = complete(f, kernelleft, kernelright, kernelup, kerneldown)[0, :, :, 0]
a = np.cos(Xa) + np.exp(Ya)
print(a[0:5, -1])
print(g[0:5, -1])
