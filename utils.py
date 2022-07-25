import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants


def tf_simp(y, axis=-2, dx=Constants.DX, rank=4):
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice3 = [slice(None)] * nd
    slice1[axis] = slice(2, None, 2)
    slice2[axis] = slice(1, -1, 2)
    slice3[axis] = slice(None, -2, 2)
    ret = tf.math.reduce_sum(dx * (y[tuple(slice1)] + 4 * y[tuple(slice2)] + y[tuple(slice3)]) / 3.0, axis=axis)
    if y.shape[axis] % 2 == 0:
        slice1[axis] = slice(-1, None, None)
        slice2[axis] = slice(-2, -1, None)
        ret += tf.math.reduce_sum(dx * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)

    return ret


def complete(H, kernelleft, kernelright, kernelup, kerneldown):
    rowup = tf.nn.conv2d(H, kernelup, strides=1, padding='VALID')
    rowdown = tf.nn.conv2d(H, kerneldown, strides=1, padding='VALID')
    a = tf.concat([rowup, H, rowdown], axis=1)
    colleft = tf.nn.conv2d(a, kernelleft, strides=1, padding='VALID')
    colright = tf.nn.conv2d(a, kernelright, strides=1, padding='VALID')
    return tf.concat([colleft, a, colright], axis=2)


def tf_diff(y, axis, rank=4):
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    ret = y[tuple(slice1)] - y[tuple(slice2)]
    return ret


def tf_simp3(y, axis=-2, dx=Constants.DX, rank=4):
    assert y.shape[axis] % 2 != 0
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice3 = [slice(None)] * nd
    slice1[axis] = slice(2, None, 2)
    slice2[axis] = slice(1, -1, 2)
    slice3[axis] = slice(None, -2, 2)
    return tf.math.reduce_sum(dx * (y[tuple(slice1)] + 4 * y[tuple(slice2)] + y[tuple(slice3)]) / 3.0, axis=axis)


def tf_simp4(y, axis=-2, dx=Constants.DX, rank=4):
    assert (y.shape[axis] - 1) % 3 == 0
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice3 = [slice(None)] * nd
    slice4 = [slice(None)] * nd
    slice1[axis] = slice(6, -3, 3)
    slice2[axis] = slice(5, -4, 3)
    slice3[axis] = slice(4, -5, 3)
    slice4[axis] = slice(3, -6, 3)
    ret = tf.math.reduce_sum(
        (3 / 8) * dx * (y[tuple(slice1)] + 3 * y[tuple(slice2)] + 3 * y[tuple(slice3)] + y[tuple(slice4)]), axis=axis)

    slice1[axis] = slice(3, 4, 1)
    slice2[axis] = slice(2, 3, 1)
    slice3[axis] = slice(1, 2, 1)
    slice4[axis] = slice(0, 1, 1)
    ret += tf.math.reduce_sum(
        dx * (5 / 192) * (
                13 * y[tuple(slice1)] + 50 * y[tuple(slice2)] + 25 * y[tuple(slice3)] + 8 * y[tuple(slice4)]),
        axis=axis)
    slice1[axis] = slice(-1, None, 1)
    slice2[axis] = slice(-2, -1, 1)
    slice3[axis] = slice(-3, -2, 1)
    slice4[axis] = slice(-4, -3, 1)
    ret += tf.math.reduce_sum(
        dx * (5 / 192) * (
                13 * y[tuple(slice1)] + 50 * y[tuple(slice2)] + 25 * y[tuple(slice3)] + 8 * y[tuple(slice4)]),
        axis=axis)
    return ret


def amper(E, Hx, Hy, beta, delta):
    pad1 = pad_function([2, 2, 2, 2])
    pad5 = pad_function([Constants.N - 2, 1, 2, 2])
    pad6 = pad_function([2, 2, 1, Constants.N - 2])
    pad7 = pad_function([2, 2, Constants.N - 2, 1])
    pad4 = pad_function([1, Constants.N - 2, 2, 2])

    x1 = tf.math.multiply(beta, Dx(Hy, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(Hy, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(Hy, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

    s1 = tf.pad(x1 + x2 + x3, pad1) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_FORWARD, perm=[1, 0, 2, 3])), Constants.PADY_FORWARD) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADY_BACWARD) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_UP, perm=[1, 0, 2, 3])), pad6) + \
         tf.pad(Dx(Hy, tf.transpose(Constants.FOURTH_DOWN, perm=[1, 0, 2, 3])), pad7)

    x1 = tf.math.multiply(beta, Dy(Hx, Constants.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(Hx, Constants.FILTER_DELTA))
    x3 = Dy(Hx, Constants.FILTER_YEE)

    s2 = tf.pad(x1 + x2 + x3, pad1) + \
         tf.pad(Dy(Hx, Constants.KERNEL_FORWARD), Constants.PADX_FORWARD) + \
         tf.pad(Dy(Hx, Constants.KERNEL_BACKWARD), Constants.PADX_BACWARD) + \
         tf.pad(Dy(Hx, Constants.FOURTH_UP), pad4) + \
         tf.pad(Dy(Hx, Constants.FOURTH_DOWN), pad5)
    return E + (Constants.DT / Constants.DX) * (s1 - s2)


def faraday(E, Hx, Hy, beta, delta):
    pad2 = pad_function([0, 0, 1, 1])
    pad3 = pad_function([1, 1, 0, 0])

    x1 = tf.math.multiply(beta, Dy(E, Constants.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(E, Constants.FILTER_DELTA))
    x3 = Dy(E, Constants.FILTER_YEE)

    s3 = tf.pad(x1 + x2 + x3, pad2) + \
         tf.pad(Dy(E, Constants.KERNEL_E_FORWARD), Constants.PADEX_FORWARD)[:, 1:-1, :, :] + \
         tf.pad(Dy(E, Constants.KERNEL_E_BACKWARD), Constants.PADEX_BACKWARD)[:, 1:-1, :, :]

    x1 = tf.math.multiply(beta, Dx(E, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(E, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(E, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

    s4 = tf.pad(x1 + x2 + x3, pad3) + \
         tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])),
                Constants.PADEY_FORWARD)[:, :, 1:-1, :] + \
         tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])),
                Constants.PADEY_BACKWARD)[:, :, 1:-1, :]

    return Hx - (Constants.DT / Constants.DX) * s3, Hy + (Constants.DT / Constants.DX) * s4


def Dy(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


# return tf.cast(tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), tf.cast(kernel,tf.dtypes.float64), strides=1, padding='VALID'), Constants.DTYPE)


def Dx(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


# return tf.cast(tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), tf.cast(kernel,tf.dtypes.float64), strides=1, padding='VALID'), Constants.DTYPE)


def f_a(c, n, k1, k2):
    e = np.cos(c * n * Constants.DT) * (
            np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) +
            np.sin(Constants.PI * k2 * Constants.X) * np.sin(
        Constants.PI * k1 * Constants.Y))

    hx = (1 / c) * np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
        Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

    hy = (1 / c) * np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

    if k1 == k2:
        energy = 1
    else:
        energy = 1 / 2

    return e, hx[1:-1, :-1], hy[:-1, 1:-1], energy


def pad_function(input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])


def loss_yee(name, beta, delta, E1, Hx1, Hy1, e_true, hx_true, hy_true, i):
    l = 0.
    for n in range(Constants.TIME_STEPS - 1):
        E1 = amper(E1, Hx1, Hy1, 0., 0.)
        Hx1, Hy1 = faraday(E1, Hx1, Hy1, beta, delta)

        l += tf.reduce_max(abs(E1[0, :, :, 0] - e_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hx1[0, :, :, 0] - hx_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0]))

    return l / (3 * (Constants.TIME_STEPS - 1))


def loss_model(model, E1, Hx1, Hy1, e_true, hx_true, hy_true, i):
    l = 0.
    for n in range(Constants.TIME_STEPS - 1):
        output = model([E1, Hx1, Hy1])
        E1 = output[0]
        Hx1 = output[1]
        Hy1 = output[2]

        l += tf.reduce_max(abs(E1[0, :, :, 0] - e_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hx1[0, :, :, 0] - hx_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0])) + \
             tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[i * Constants.TIME_STEPS + (n + 1), :, :, 0]))
    return l / (3 * (Constants.TIME_STEPS - 1))


def custom_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return tf.math.reduce_mean(abs(y_true[7:-7,7:-7] - y_pred[7:-7,7:-7])) / Constants.DT
    # return tf.math.reduce_mean(abs(y_true[:,5:-5,5:-5,] - y_pred[:,5:-5,5:-5,:])) / Constants.DT


def custom_loss3(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return tf.math.reduce_mean(abs(y_true[7:-7,7:-7] - y_pred[7:-7,7:-7]))


class DRP_LAYER(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.pars1 = tf.Variable(0., trainable=False, dtype=Constants.DTYPE, name='beta')
        self.pars2 = tf.Variable(-0.125, trainable=True, dtype=Constants.DTYPE, name='delta')
        self.pars3 = tf.Variable(0., trainable=False, dtype=Constants.DTYPE, name='zero')

    def call(self, input):

        E, Hx, Hy = input
        E_n = amper(E, Hx, Hy, self.pars1, self.pars2)
        Hx_n, Hy_n = faraday(E_n, Hx, Hy, self.pars1, self.pars2)

        E_m = amper(E_n, Hx_n, Hy_n, self.pars1, self.pars2)
        Hx_m, Hy_m = faraday(E_m, Hx_n, Hy_n, self.pars1, self.pars2)

        # hx = complete(Hx_n, Constants.KLEFT, Constants.KRIGHT, Constants.KUP, Constants.KDOWN)

        # hy = complete(Hy_n, tf.transpose(Constants.KUP, [1, 0, 2, 3]), tf.transpose(Constants.KDOWN, [1, 0, 2, 3]),
        #           tf.transpose(Constants.KLEFT, [1, 0, 2, 3]), tf.transpose(Constants.KRIGHT, [1, 0, 2, 3]))

        # inte = tf_simp3(tf_simp3(E_n ** 2, rank=4), rank=3)
        # inthx = tf_simp3(tf_simp4(hx ** 2, rank=4), rank=3)
        # inthy = tf_simp4(tf_simp3(hy ** 2, rank=4), rank=3)

        #y1 = tf.math.multiply(self.pars1, Dy(E_n, Constants.FILTER_BETA))
        #y2 = tf.math.multiply(self.pars2, Dy(E_n, Constants.FILTER_DELTA))
        #y3 = Dy(E_n, Constants.FILTER_YEE)
        #dEdy=Dx(y1+y1+y2, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

        #x1 = tf.math.multiply(self.pars1, Dx(E_n, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
        #x2 = tf.math.multiply(self.pars2, Dx(E_n, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
        #x3 = Dx(E_n, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))
        #dEdx=Dy(x1+x2+x3, Constants.FILTER_YEE)




        #divergence = ( dhydy[:,1:-1,:,:]+ dhxdx[:, :,  1:-1, :])/Constants.DX
        #divergence = (dhydy[:, 1:-1, :, :] + dhxdx[:, :, 1:-1, :]) / Constants.DX

        #divergence = (tf_diff(Hy_n[:, 1:-1, :, :], axis=2) + tf_diff(Hx_n[:, :, 1:-1, :], axis=1))
        #divergence=dEdx-dEdy

        return E_n, Hx_n, Hy_n, E_m, Hx_m, Hy_m
