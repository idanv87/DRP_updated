import math

import numpy as np
from dataclasses import dataclass

import tensorflow as tf



@dataclass(frozen=True)
class Constants:
    FROG = 2
    FROG1 = 2
    N = 110
    PI = math.pi
    YMIN, YMAX = 0.0, 1.0
    XMIN, XMAX = 0.0, 1.0
    Z = 1.
    T = 0.01
    TIME_STEPS = 400
    DT = T / TIME_STEPS
    LX = XMAX - XMIN
    LY = YMAX - YMIN
    DX = LX / (N - 1)
    DY = LY / (N - 1)

    X, Y = np.meshgrid(np.linspace(0., XMAX, N), np.linspace(0., XMAX, N), indexing='ij')
    K1_TRAIN = [1.,3.]
    K2_TRAIN = [1.,3.]
    K1_TEST = [1.]
    K2_TEST = [1.]

    PADX_FORWARD = tf.constant([[0, 0], [1, 1], [1, N - 2], [0, 0]], shape=[4, 2])
    PADX_BACWARD = tf.constant([[0, 0], [1, 1], [N - 2, 1], [0, 0]], shape=[4, 2])
    PADY_FORWARD = tf.constant([[0, 0], [1, N - 2], [1, 1], [0, 0]], shape=[4, 2])
    PADY_BACWARD = tf.constant([[0, 0], [N - 2, 1], [1, 1], [0, 0]], shape=[4, 2])

    A = np.array([-23 / 24, 7 / 8, 1 / 8, -1 / 24, 0.]).reshape(1, 5)
    #A=np.array([-1.,1.,0.,0.,0.]).reshape(1, 5)
    B = np.zeros((1, N - 5 - 1))

    KERNEL_FORWARD = (1 / DX) * tf.cast(np.append(A, B).reshape(1, N - 1, 1, 1), tf.dtypes.float64)
    KERNEL_BACKWARD = -tf.reverse(KERNEL_FORWARD, [1])

    PADEX_FORWARD = tf.constant([[0, 0], [0, 0], [0, N - 2], [0, 0]], shape=[4, 2])
    PADEX_BACKWARD = tf.constant([[0, 0], [0, 0], [N - 2, 0], [0, 0]], shape=[4, 2])
    PADEY_FORWARD = tf.constant([[0, 0], [0, N - 2], [0, 0], [0, 0]], shape=[4, 2])
    PADEY_BACKWARD = tf.constant([[0, 0], [N - 2, 0], [0, 0], [0, 0]], shape=[4, 2])

    D = np.zeros((1, N - 5))
    KERNEL_E_FORWARD = (1 / DX) * tf.cast(np.append(A, D).reshape(1, N, 1, 1), tf.dtypes.float64)
    KERNEL_E_BACKWARD = -tf.reverse(KERNEL_E_FORWARD, [1])

    FILTER1 = tf.constant([[0., 0., 0., 0.], [1 / (3 * DX), -1 / DX, 1 / DX, -1 / (3 * DX)], [0., 0., 0., 0.]],
                          shape=[3, 4, 1, 1],dtype=tf.dtypes.float64)
    FILTER2 = tf.constant([[0., 0., 0., 0.], [-1 / (3 * DX), 0, 0, 1 / (3 * DX)], [0., 0., 0., 0.]], shape=[3, 4, 1, 1],dtype=tf.dtypes.float64)

    # FILTER3 = tf.constant([[0., 0., 0., 0.], [0, -1, 1, 0], [0., 0., 0., 0.]], shape=[3, 4, 1, 1])

    PADUP = tf.constant([[0, N - 3], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
    PADDOWN = tf.constant([[N - 3, 0], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
    FOURTH_UP = tf.pad(tf.constant([1 / (24 * DX), -9 / (8 * DX), 9 / (8 * DX), -1 / (24 * DX)], shape=[1, 4, 1, 1],dtype=tf.dtypes.float64), PADUP)
    FOURTH_DOWN = tf.pad(tf.constant([1 / (24 * DX), -9 / (8 * DX), 9 / (8 * DX), -1 / (24 * DX)], shape=[1, 4, 1, 1],dtype=tf.dtypes.float64), PADDOWN)
    #FOURTH_UP = tf.pad(tf.constant([0, -1/DX, 1/DX, 0], shape=[1, 4, 1, 1],dtype=tf.dtypes.float64),
                     #  PADUP)
    #FOURTH_DOWN = tf.pad(tf.constant([0, -1/DX, 1/DX, 0], shape=[1, 4, 1, 1],dtype=tf.dtypes.float64),
                      #   PADDOWN)