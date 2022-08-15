import math

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    PATH = '/Users/idanversano/documents/pycharm/files/'
    # PATH = '/home/ubuntu/files/'

    DTYPE = tf.dtypes.float64

    N = 31
    T = 0.01
    TIME_STEPS = 26
    CROSS_VAL = 1
    EPOCHS = 600
    BATCH_SIZE = 128
    K1_TRAIN = np.arange(1, 61,2)
    K2_TRAIN = np.arange(1, 61,2)

    K1_DRP = np.arange(15, 31, 1)
    K2_DRP = np.arange(15, 31, 1)

    # K1_VAL = np.arange(1, 41, 5)
    # K2_VAL = np.arange(1, 41, 5)

    TRAIN_NUM = 20
    K1_TEST = np.arange(8, 45, 5)
    K2_TEST = np.arange(21, 22, 1)

    #assert N % 2 != 0
    #assert (N) % 3 == 0

    PI = math.pi
    YMIN, YMAX = 0.0, 1.
    XMIN, XMAX = 0.0, 1.

    DT = T / (TIME_STEPS-1 )

    LX = XMAX - XMIN
    LY = YMAX - YMIN
    DX = LX / (N - 1)
    DY = LY / (N - 1)

    CFL = DT / DX

    X1 = np.linspace(0., XMAX, N)
    X2 = np.linspace(0., XMAX, N)

    # plt.plot(np.cos(5*math.pi*X1))
    # plt.show()
    X, Y = np.meshgrid(X1, X2, indexing='ij')

    K1_VAL = [1., 2]
    K2_VAL = [1.]



    TEST_NUM = len(K1_TEST) * len(K2_TEST)

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

    A = np.append(np.array([[35 / 16, -35 / 16, 21 / 16, -5 / 16]]), np.zeros((1, N - 5)))
    B = np.append(np.array([[4, -6, 4, -1]]), np.zeros((1, N - 6)))

    KLEFT = np.reshape(A, [1, A.shape[0], 1, 1])
    KRIGHT = tf.reverse(KLEFT, [1])

    KUP = np.reshape(B, [B.shape[0], 1, 1, 1])
    KDOWN = tf.reverse(KUP, [0])
