import math

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataclasses import dataclass


# @dataclass(frozen=True)
class Constants:
    PATH = '/Users/idanversano/documents/pycharm/files/'
    FIGURES_PATH = '/Users/idanversano/documents/papers/drp/figures/'
    # PATH = '/home/ubuntu/files/'

    DTYPE = tf.dtypes.float64

    PI = math.pi

    CROSS_VAL = 1
    EPOCHS = 600
    BATCH_SIZE = 128
    # K1_TRAIN = np.arange(11,20, 1)
    # K2_TRAIN = np.arange(11, 20, 1)
    #
    K1_TRAIN = np.array([11,13,17,19])
    K2_TRAIN = np.array([11,13,17,19])

    K1_DRP = np.arange(11, 20, 1)  # should be N
    K2_DRP = np.arange(11, 20, 1)

    '''
    4 filters below define the derivative kernel as function of beta,gamma,delta:
    DX=beta*fiter_beta+...+1*filter_yee
    '''

    FILTER_BETA = tf.constant([[0., -1, 1, 0.], [0, 2, -2, 0], [0., -1, 1, 0.]],
                              shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_DELTA = tf.constant([[0., 0, 0, 0.], [-1, 3, -3, 1], [0., 0., 0., 0.]],
                               shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_GAMMA = tf.constant([[-1, 0, 0, 1], [0., 6, -6, 0.], [-1, 0., 0., 1]],
                               shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_YEE = tf.constant([[0., 0, 0, 0.], [0, -1, 1, 0], [0., 0., 0., 0.]],
                             shape=[3, 4, 1, 1], dtype=DTYPE)

    # CENTRAL = tf.constant([-1, 1], shape=[1, 2, 1, 1], dtype=DTYPE)
    '''
    A is one sided derivative of order 4 kernel of 5 points used for the non-compact stencil
    we use
    '''
    A = np.array([-23 / 24, 7 / 8, 1 / 8, -1 / 24, 0.]).reshape(1, 5)

    def __init__(self, n, x, t, time_steps, k1_test, k2_test):
        self.XMAX = x
        self.N = n
        self.TIME_STEPS = time_steps
        self.T = t
        self.K1_TEST = k1_test
        self.K2_TEST = k2_test
        # self.TEST_NUM = len(self.K1_TEST) * len(self.K2_TEST)

        self.DT = self.T / (self.TIME_STEPS - 1)
        self.DX = self.XMAX / (self.N - 1)
        self.DY = self.XMAX / (self.N - 1)
        self.CFL = self.DT / self.DX
        self.X1 = np.linspace(0., self.XMAX, self.N)
        self.X2 = np.linspace(0., self.XMAX, self.N)
        self.X, self.Y = np.meshgrid(self.X1, self.X2, indexing='ij')
        '''
        4 tesnors below are padding kernels required to pad  rows/cols after covolution with 
        derivative
        '''
        self.PADX_FORWARD = tf.constant([[0, 0], [1, 1], [1, self.N - 2], [0, 0]], shape=[4, 2])
        self.PADX_BACWARD = tf.constant([[0, 0], [1, 1], [self.N - 2, 1], [0, 0]], shape=[4, 2])
        self.PADY_FORWARD = tf.constant([[0, 0], [1, self.N - 2], [1, 1], [0, 0]], shape=[4, 2])
        self.PADY_BACWARD = tf.constant([[0, 0], [self.N - 2, 1], [1, 1], [0, 0]], shape=[4, 2])

        '''
        4 tesnors below with the names "kernel_| are derivative kernels for the boundaries 
        which are added after the covolution is calculated in  the interior points
        (in tensor flow one cannot assign tensor to tensor and thats why we:
         1.calculate derivtive for inner point)
         2. pad with zeros 
         3. calculate the one sided derivative for 2 columns on each side (kernels forward and backwards) and pad
         4. calculate for two remaining rows with kernels fourth up and down (to avoid overlapping with step 3 we omit the 
         first two and last entries)
         two terms in each row so central derivative of order fourth is enough )
        '''
        self.B = np.zeros((1, self.N - 5 - 1))
        self.KERNEL_FORWARD = tf.cast(np.append(Constants.A, self.B).reshape(1, self.N - 1, 1, 1), Constants.DTYPE)
        self.KERNEL_BACKWARD = -tf.reverse(self.KERNEL_FORWARD, [1])
        self.PADEX_FORWARD = tf.constant([[0, 0], [0, 0], [0, self.N - 2], [0, 0]], shape=[4, 2])
        self.PADEX_BACKWARD = tf.constant([[0, 0], [0, 0], [self.N - 2, 0], [0, 0]], shape=[4, 2])
        self.PADEY_FORWARD = tf.constant([[0, 0], [0, self.N - 2], [0, 0], [0, 0]], shape=[4, 2])
        self.PADEY_BACKWARD = tf.constant([[0, 0], [self.N - 2, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.D = np.zeros((1, self.N - 5))
        self.KERNEL_E_FORWARD = tf.cast(np.append(Constants.A, self.D).reshape(1, self.N, 1, 1), Constants.DTYPE)
        self.KERNEL_E_BACKWARD = -tf.reverse(self.KERNEL_E_FORWARD, [1])
        self.PADUP = tf.constant([[0, self.N - 3], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.PADDOWN = tf.constant([[self.N - 3, 0], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.FOURTH_UP = tf.pad(
            tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=Constants.DTYPE),
            self.PADUP)
        self.FOURTH_DOWN = tf.pad(
            tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=Constants.DTYPE),
            self.PADDOWN)

        self.H = np.append(np.array([[35 / 16, -35 / 16, 21 / 16, -5 / 16]]), np.zeros((1, self.N - 5)))
        self.G = np.append(np.array([[4, -6, 4, -1]]), np.zeros((1, self.N - 6)))

        self.KLEFT = np.reshape(self.H, [1, self.H.shape[0], 1, 1])
        self.KRIGHT = tf.reverse(self.KLEFT, [1])

        self.KUP = np.reshape(self.G, [self.G.shape[0], 1, 1, 1])
        self.KDOWN = tf.reverse(self.KUP, [0])


model_constants = Constants(21, 1, 2 / (5 * 2 ** 0.5), 73, 1., 1.)
