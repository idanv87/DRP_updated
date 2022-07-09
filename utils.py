import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants



def amper(E, Hx, Hy,par):
        pad1 = pad_function([2, 2, 2, 2])
        pad5 = pad_function([Constants.N - 2, 1, 2, 2])
        pad6 = pad_function([2, 2, 1, Constants.N - 2])
        pad7 = pad_function([2, 2, Constants.N - 2, 1])
        pad4 = pad_function([1, Constants.N - 2, 2, 2])

        s1 = tf.pad(par * Dx(Hy, tf.transpose(Constants.FILTER1, perm=[1, 0, 2, 3])) + \
                    Dx(Hy, tf.transpose(Constants.FILTER2, perm=[1, 0, 2, 3])), pad1) + \
             tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_FORWARD, perm=[1, 0, 2, 3])), Constants.PADY_FORWARD) + \
             tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADY_BACWARD) + \
             tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_UP, perm=[1, 0, 2, 3])), pad6) + \
             tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_DOWN, perm=[1, 0, 2, 3])), pad7)

        s2 = tf.pad(par * Dy(Hx, Constants.FILTER1) + Dy(Hx, Constants.FILTER2), pad1) + \
             tf.pad(Dy(Hx, Constants.KERNEL_FORWARD), Constants.PADX_FORWARD) + \
             tf.pad(Dy(Hx, Constants.KERNEL_BACKWARD), Constants.PADX_BACWARD) + \
             tf.pad(Dy(Hx, Constants.FOURTH_UP), pad4) + \
             tf.pad(Dy(Hx, Constants.FOURTH_DOWN), pad5)
        return E + Constants.DT * (s1 - s2)

def faraday( E, Hx, Hy,par):
        pad2 = pad_function([0, 0, 1, 1])
        pad3 = pad_function([1, 1, 0, 0])

        s3 = tf.pad(par * Dy(E, Constants.FILTER1) + Dy(E, Constants.FILTER2), pad2) + \
             tf.pad(Dy(E, Constants.KERNEL_E_FORWARD), Constants.PADEX_FORWARD)[:, 1:-1, :, :] + \
             tf.pad(Dy(E, Constants.KERNEL_E_BACKWARD), Constants.PADEX_BACKWARD)[:, 1:-1, :, :]

        s4 = tf.pad(par * Dx(E, tf.transpose(Constants.FILTER1, perm=[1, 0, 2, 3])) + \
                    Dx(E, tf.transpose(Constants.FILTER2, perm=[1, 0, 2, 3])), pad3) + \
             tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])), Constants.PADEY_FORWARD)[:,
             :, 1:-1, :] + \
             tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADEY_BACKWARD)[
             :, :, 1:-1, :]

        return Hx - Constants.DT * s3, Hy + Constants.DT * s4

def Dy(B, kernel):
        return tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), kernel, strides=1, padding='VALID')

def Dx( B, kernel):
        return tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), kernel, strides=1, padding='VALID')


def f_a(c, n, k1, k2):
    e = c * np.cos(c * n * Constants.DT) * (
            np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) +
            np.sin(Constants.PI * k2 * Constants.X) * np.sin(
        Constants.PI * k1 * Constants.Y))

    hx = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
        Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

    hy = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

    return e, hx[1:-1, :-1], hy[:-1, 1:-1]

def pad_function( input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])

class MAIN_LAYER(keras.layers.Layer):

    def __init__(self):
        super(MAIN_LAYER, self).__init__()
        self.pars = tf.Variable(2., trainable=True, dtype=tf.dtypes.float64, name='w')



    def call(self, input):
        E, Hx, Hy = input
        E_n = amper(tf.cast(E,tf.dtypes.float64), tf.cast(Hx,tf.dtypes.float64), tf.cast(Hy,tf.dtypes.float64), self.pars)
        Hx_n, Hy_n = faraday(tf.cast(E_n,tf.dtypes.float64), tf.cast(Hx,tf.dtypes.float64), tf.cast(Hy,tf.dtypes.float64), self.pars)
        E_m = amper(tf.cast(E_n,tf.dtypes.float64), tf.cast(Hx_n,tf.dtypes.float64), tf.cast(Hy_n,tf.dtypes.float64), self.pars)
        Hx_m, Hy_m = faraday(E_m, Hx_n, Hy_n, self.pars)
        return tf.concat([E_n, E_m], 1), tf.concat([Hx_n, Hx_m], 1), tf.concat([Hy_n, Hy_m], 1)