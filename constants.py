import math

import tensorflow as tf

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    PATH = '/Users/idanversano/documents/pycharm/files/'
    #PATH = '/home/ubuntu/files/'

    DTYPE = tf.dtypes.float64

    N = 160
    PI = math.pi
    YMIN, YMAX = 0.0, 1.0
    XMIN, XMAX = 0.0, 1.0

    T = 1
    TIME_STEPS = 1000
    DT = T / TIME_STEPS
    LX = XMAX - XMIN
    LY = YMAX - YMIN
    DX = LX / (N - 1)
    DY = LY / (N - 1)

    X1 = np.linspace(0., XMAX, N)
    X2 = np.linspace(0., XMAX, N)

    X, Y = np.meshgrid(X1, X2, indexing='ij')
    K1_TRAIN = [1.]
    K2_TRAIN = [1.]

    K1_VAL = [2.]
    K2_VAL = [2.]

    K1_TEST = [2, 4, 6, 8, 10, 12]
    K2_TEST = [1]

    PADX_FORWARD = tf.constant([[0, 0], [1, 1], [1, N - 2], [0, 0]], shape=[4, 2])
    PADX_BACWARD = tf.constant([[0, 0], [1, 1], [N - 2, 1], [0, 0]], shape=[4, 2])
    PADY_FORWARD = tf.constant([[0, 0], [1, N - 2], [1, 1], [0, 0]], shape=[4, 2])
    PADY_BACWARD = tf.constant([[0, 0], [N - 2, 1], [1, 1], [0, 0]], shape=[4, 2])

    A = np.array([-23 / 24, 7 / 8, 1 / 8, -1 / 24, 0.]).reshape(1, 5)

    B = np.zeros((1, N - 5 - 1))

    KERNEL_FORWARD = tf.cast(np.append(A, B).reshape(1, N - 1, 1, 1), DTYPE)

    KERNEL_BACKWARD = -tf.reverse(KERNEL_FORWARD, [1])

    PADEX_FORWARD = tf.constant([[0, 0], [0, 0], [0, N - 2], [0, 0]], shape=[4, 2])
    PADEX_BACKWARD = tf.constant([[0, 0], [0, 0], [N - 2, 0], [0, 0]], shape=[4, 2])
    PADEY_FORWARD = tf.constant([[0, 0], [0, N - 2], [0, 0], [0, 0]], shape=[4, 2])
    PADEY_BACKWARD = tf.constant([[0, 0], [N - 2, 0], [0, 0], [0, 0]], shape=[4, 2])

    D = np.zeros((1, N - 5))
    KERNEL_E_FORWARD = tf.cast(np.append(A, D).reshape(1, N, 1, 1), DTYPE)
    KERNEL_E_BACKWARD = -tf.reverse(KERNEL_E_FORWARD, [1])

    FILTER_BETA = tf.constant([[0., -1, 1, 0.], [0, 2, -2, 0], [0., -1, 1, 0.]],
                              shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_DELTA = tf.constant([[0., 0, 0, 0.], [-1, 3, -3, 1], [0., 0., 0., 0.]],
                               shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_YEE = tf.constant([[0., 0, 0, 0.], [0, -1, 1, 0], [0., 0., 0., 0.]],
                             shape=[3, 4, 1, 1], dtype=DTYPE)

    PADUP = tf.constant([[0, N - 3], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
    PADDOWN = tf.constant([[N - 3, 0], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
    FOURTH_UP = tf.pad(
        tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=DTYPE),
        PADUP)
    FOURTH_DOWN = tf.pad(
        tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=DTYPE),
        PADDOWN)

    CENTRAL = tf.constant([-1, 1], shape=[1, 2, 1, 1], dtype=DTYPE)
