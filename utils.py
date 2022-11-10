import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.linalg import polar

from DRP_multiple_networks.constants import model_constants
from DRP_multiple_networks.auxilary.aux_functions import relative_norm
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2


class DRP_LAYER(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.pars1 = tf.Variable(0., trainable=True, dtype=model_constants.DTYPE, name='beta')
        self.pars2 = tf.Variable(-0.09, trainable=True, dtype=model_constants.DTYPE, name='delta')
        self.pars3 = tf.Variable(0., trainable=True, dtype=model_constants.DTYPE, name='gamma')

    def call(self, input):
        E1_true, Hx1_true, Hy1_true, E2_true, Hx2_true, Hy2_true, E3_true, Hx3_true, Hy3_true = input

        E_2 = amper(E1_true, Hx1_true, Hy1_true, self.pars1, self.pars2, self.pars3, model_constants)
        Hx_2, Hy_2 = faraday(E_2, Hx1_true, Hy1_true, self.pars1, self.pars2, self.pars3, model_constants)

        E_3 = amper(E_2, Hx_2, Hy_2, self.pars1, self.pars2, self.pars3, model_constants)
        Hx_3, Hy_3 = faraday(E_3, Hx_2, Hy_2, self.pars1, self.pars2, self.pars3, model_constants)

        E_4 = amper(E_3, Hx_3, Hy_3, self.pars1, self.pars2, self.pars3, model_constants)
        Hx_4, Hy_4 = faraday(E_4, Hx_3, Hy_3, self.pars1, self.pars2, self.pars3, model_constants)

        # E_2 = amper(E1, Hx1, Hy1, self.pars1, (16*self.pars1-1)/24, -self.pars1/3, model_constants)
        # Hx_2, Hy_2 = faraday(E2, Hx1, Hy1, self.pars1, (16*self.pars1-1)/24, -self.pars1/3, model_constants)
        #
        # E_3 = amper(E_2, Hx_2, Hy_2, self.pars1, (16*self.pars1-1)/24, -self.pars1/3, model_constants)
        # Hx_3, Hy_3 = faraday(E_3, Hx_2, Hy_2, self.pars1, (16*self.pars1-1)/24, -self.pars1/3, model_constants)
        #
        # E_4 = amper(E_3, Hx_3, Hy_3, self.pars1, (16*self.pars1-1)/24, -self.pars1/3, model_constants)
        # Hx_4, Hy_4 = faraday(E_4, Hx_3, Hy_3, self.pars1,(16*self.pars1-1)/24, -self.pars1/3, model_constants)

        # hx = complete(Hx_n, C.KLEFT, C.KRIGHT, C.KUP, C.KDOWN)

        # hy = complete(Hy_n, tf.transpose(C.KUP, [1, 0, 2, 3]), tf.transpose(C.KDOWN, [1, 0, 2, 3]),
        #           tf.transpose(C.KLEFT, [1, 0, 2, 3]), tf.transpose(C.KRIGHT, [1, 0, 2, 3]))

        # inte = tf_simp3(tf_simp3(E_n ** 2, rank=4), rank=3)
        # inthx = tf_simp3(tf_simp4(hx ** 2, rank=4), rank=3)
        # inthy = tf_simp4(tf_simp3(hy ** 2, rank=4), rank=3)

        # y1 = tf.math.multiply(self.pars1, Dy(E_n, C.FILTER_BETA))
        # y2 = tf.math.multiply(self.pars2, Dy(E_n, C.FILTER_DELTA))
        # y3 = Dy(E_n, C.FILTER_YEE)
        # dEdy=Dx(y1+y1+y2, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))

        # x1 = tf.math.multiply(self.pars1, Dx(E_n, tf.transpose(C.FILTER_BETA, perm=[1, 0, 2, 3])))
        # x2 = tf.math.multiply(self.pars2, Dx(E_n, tf.transpose(C.FILTER_DELTA, perm=[1, 0, 2, 3])))
        # x3 = Dx(E_n, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))
        # dEdx=Dy(x1+x2+x3, C.FILTER_YEE)

        # divergence = ( dhydy[:,1:-1,:,:]+ dhxdx[:, :,  1:-1, :])/C.DX
        # divergence = (dhydy[:, 1:-1, :, :] + dhxdx[:, :, 1:-1, :]) / C.DX

        # divergence = (tf_diff(Hy_n[:, 1:-1, :, :], axis=2) + tf_diff(Hx_n[:, :, 1:-1, :], axis=1))
        # divergence=dEdx-dEdy
        # d = tf_simp3(tf_simp3(E1 ** 2, rank=4), rank=3) * 0 + drp_loss(self.pars2)

        # d = drp_loss(self.pars2)

        return E_2, Hx_2, Hy_2, E_3, Hx_3, Hy_3, E_4, Hx_4, Hy_4


def loss_yee(name, beta, delta, gamma, test_data, C, norm='polar'):
    """'
    this function recieve analytic solution, solve the equation and compare it to analytical solution
    at each time step.
    The output is the average error in some norm
    """

    assert norm in ['l2', 'polar']

    E = np.expand_dims(test_data['e'][0], axis=(0, -1))
    Hx = np.expand_dims(test_data['hx'][0], axis=(0, -1))
    Hy = np.expand_dims(test_data['hy'][0], axis=(0, -1))

    error = 0.
    for n in range(C.TIME_STEPS - 1):
        E = amper(E, Hx, Hy, beta, delta, gamma, C)
        Hx, Hy = faraday(E, Hx, Hy, beta, delta, gamma, C)
        #    plt.plot(E1[0,:,10,0],'-')
        #    plt.plot(test_data['e'][n + 1][:,10])
        #    plt.show()
        # print(q)

        if norm == 'l2':
            error += relative_norm(E[0, :, :, 0], test_data['e'][n + 1]) + \
                     relative_norm(Hx[0, :, :, 0], test_data['hx'][n + 1]) + \
                     relative_norm(Hy[0, :, :, 0], test_data['hy'][n + 1])
        else:
            error += np.mean(abs(polar(E[0, :, :, 0])[0] - polar(test_data['e'][n + 1])[0]) ** 2) + \
                     np.mean(abs(polar(Hx[0, :, :, 0])[0] - polar(test_data['hx'][n + 1])[0]) ** 2) + \
                     np.mean(abs(polar(Hy[0, :, :, 0])[0] - polar(test_data['hy'][n + 1])[0]) ** 2)

    return error / (3 * (C.TIME_STEPS - 1))


def amper(E, Hx, Hy, beta, delta, gamma, C):
    cfl = C.CFL
    pad1 = pad_function([2, 2, 2, 2])
    pad5 = pad_function([C.N - 2, 1, 2, 2])
    pad6 = pad_function([2, 2, 1, C.N - 2])
    pad7 = pad_function([2, 2, C.N - 2, 1])
    pad4 = pad_function([1, C.N - 2, 2, 2])

    x1 = tf.math.multiply(beta, Dx(Hy, tf.transpose(C.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(Hy, tf.transpose(C.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(Hy, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))
    x4 = tf.math.multiply(gamma, Dx(Hy, tf.transpose(C.FILTER_GAMMA, perm=[1, 0, 2, 3])))
    s1 = tf.pad(x1 + x2 + x3 + x4, pad1) + \
         tf.pad(Dx(Hy, tf.transpose(C.KERNEL_FORWARD, perm=[1, 0, 2, 3])), C.PADY_FORWARD) + \
         tf.pad(Dx(Hy, tf.transpose(C.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), C.PADY_BACWARD) + \
         tf.pad(Dx(Hy, tf.transpose(C.FOURTH_UP, perm=[1, 0, 2, 3])), pad6) + \
         tf.pad(Dx(Hy, tf.transpose(C.FOURTH_DOWN, perm=[1, 0, 2, 3])), pad7)

    x1 = tf.math.multiply(beta, Dy(Hx, C.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(Hx, C.FILTER_DELTA))
    x3 = Dy(Hx, C.FILTER_YEE)
    x4 = tf.math.multiply(gamma, Dy(Hx, C.FILTER_GAMMA))

    s2 = tf.pad(x1 + x2 + x3 + x4, pad1) + \
         tf.pad(Dy(Hx, C.KERNEL_FORWARD), C.PADX_FORWARD) + \
         tf.pad(Dy(Hx, C.KERNEL_BACKWARD), C.PADX_BACWARD) + \
         tf.pad(Dy(Hx, C.FOURTH_UP), pad4) + \
         tf.pad(Dy(Hx, C.FOURTH_DOWN), pad5)
    return E + (cfl) * (s1 - s2)


def faraday(E, Hx, Hy, beta, delta, gamma, C):
    cfl = C.CFL
    pad2 = pad_function([0, 0, 1, 1])
    pad3 = pad_function([1, 1, 0, 0])

    x1 = tf.math.multiply(beta, Dy(E, C.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(E, C.FILTER_DELTA))
    x3 = Dy(E, C.FILTER_YEE)
    x4 = tf.math.multiply(gamma, Dy(E, C.FILTER_GAMMA))

    s3 = tf.pad(x1 + x2 + x3 + x4, pad2) + \
         tf.pad(Dy(E, C.KERNEL_E_FORWARD), C.PADEX_FORWARD)[:, 1:-1, :, :] + \
         tf.pad(Dy(E, C.KERNEL_E_BACKWARD), C.PADEX_BACKWARD)[:, 1:-1, :, :]

    x1 = tf.math.multiply(beta, Dx(E, tf.transpose(C.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(E, tf.transpose(C.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(E, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))
    x4 = tf.math.multiply(gamma, Dx(E, tf.transpose(C.FILTER_GAMMA, perm=[1, 0, 2, 3])))

    s4 = tf.pad(x1 + x2 + x3 + x4, pad3) + \
         tf.pad(Dx(E, tf.transpose(C.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])),
                C.PADEY_FORWARD)[:, :, 1:-1, :] + \
         tf.pad(Dx(E, tf.transpose(C.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])),
                C.PADEY_BACKWARD)[:, :, 1:-1, :]

    return Hx - (cfl) * s3, Hy + (cfl) * s4


def Dy(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


def Dx(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


def pad_function(input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])


def custom_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # return tf.math.reduce_mean(abs(y_true[:, 5:-5, 3:-3, :] - y_pred[:, 3:-3, 3:-3, :])) / C.CFL
    return tf.math.reduce_mean(abs((y_true - y_pred) ** 2)) / model_constants.CFL
    # return tf.math.reduce_mean(abs(y_true[:,5:-5,5:-5,] - y_pred[:,5:-5,5:-5,:])) / C.DT


def custom_loss3(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # return tf.math.reduce_mean(abs(y_true[:, 5:-5, 3:-3, :] - y_pred[:, 3:-3, 3:-3, :])) / C.CFL
    return tf.math.reduce_mean(tf.math.sin(abs(y_true - y_pred))) / model_constants.CFL
    # return tf.math.reduce_mean(abs(y_true[7:-7, 7:-7] - y_pred[7:-7, 7:-7]))


def custom_loss_drp(y_true, y_pred):
    return abs(y_pred)


'''
3 functions below calculate analytic solutions to the equation:
'''


def fE(FE, m, T, omega, C):
    t = T + C.DT * m
    return np.cos(omega * t) * FE


def fHX(FHX, m, T, omega, C):
    t = T + m * C.DT / 2
    z = np.sin(omega * t) * (1 / omega) * FHX
    return z[:, 1:-1, :-1]


def fHY(FHY, m, T, omega, C):
    t = T + m * C.DT / 2
    z = np.sin(omega * t) * (1 / omega) * FHY
    return z[:, :-1, 1:-1]
