import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from DRP_multiple_networks.constants import Constants
from DRP_multiple_networks.auxilary.aux_functions import relative_norm
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2

C = Constants()

def ee(a):
    pass

def loss_yee(name, beta, delta, gamma, test_data):
    """'
    this function recieve analytic solution, solve the equation and compare it to analytical solution
    at each time step.
    The output is the average error
    """

    E1 = np.expand_dims(test_data['e'][0], axis=(0, -1))
    Hx1 = np.expand_dims(test_data['hx'][0], axis=(0, -1))
    Hy1 = np.expand_dims(test_data['hy'][0], axis=(0, -1))

    l = 0.
    for n in range(Constants.TIME_STEPS - 3):
        E1 = amper(E1, Hx1, Hy1, beta, delta, gamma)
        Hx1, Hy1 = faraday(E1, Hx1, Hy1, beta, delta, gamma)

        # l += tf.reduce_mean(abs(E1[0, :, :, 0] - test_data['e'][n + 1])) + \
        #      tf.reduce_mean(abs(Hx1[0, :, :, 0] - test_data['hx'][n + 1])) + \
        #      tf.reduce_mean(abs(Hy1[0, :, :, 0] - test_data['hy'][n + 1]))
        l += relative_norm(E1[0, :, :, 0], test_data['e'][n + 1]) + \
             relative_norm(Hx1[0, :, :, 0], test_data['hx'][n + 1]) + \
             relative_norm(Hy1[0, :, :, 0], test_data['hy'][n + 1])

    return l / (3 * (Constants.TIME_STEPS - 3))


def fd_solver(beta, delta, gamma, test_data):
    """"
    this function solve the equation for a given test data.
    and save the energy at each step
    """
    E1 = np.expand_dims(test_data['e'][0], axis=(0, -1))
    Hx1 = np.expand_dims(test_data['hx'][0], axis=(0, -1))
    Hy1 = np.expand_dims(test_data['hy'][0], axis=(0, -1))
    energy = []
    for n in range(C.TIME_STEPS - 3):
        E1 = amper(E1, Hx1, Hy1, beta, delta, gamma)
        Hx1, Hy1 = faraday(E1, Hx1, Hy1, beta, delta, gamma)
        energy.append((tf.reduce_sum(E1 ** 2) + tf.reduce_sum(Hx1 ** 2) + tf.reduce_sum(Hy1 ** 2)) * C.DX * C.DX)
    return energy


def drp_loss(a):
    """"
    this function is the loss function of the drp_function given by an integral from pi/2
    to pi over the dispersion relation minus 1 (see tex file).
    """
    X = tf.cast(tf.linspace(math.pi / 2, math.pi, 200), tf.dtypes.float64)
    x, y = tf.meshgrid(X, X, indexing='ij')

    f = ((1 - a) ** 2) * (tf.cos(3 * x) + tf.cos(3 * y)) + \
        (6 * a - 6 * a ** 2) * (tf.cos(2 * x) + tf.cos(2 * y)) + \
        (15 * a ** 2 - 6 * a) * (tf.cos(x) + tf.cos(y))
    omega_over_k = (2 / (Constants.CFL * tf.sqrt(x ** 2 + y ** 2))) * tf.asin(
        Constants.CFL * tf.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    ret = tf_trapz(tf_trapz(abs(omega_over_k - 1), axis=-1, dx=X[1] - X[0], rank=2), axis=-1, dx=X[1] - X[0], rank=1)

    return ret


def fE(FE, m, T, c):
    t = T + C.DT * m
    return np.cos(c * t) * FE


def fHX(FHX, m, T, c):
    t = T + m * C.DT / 2
    z = np.sin(c * t) * (1 / c) * FHX
    return z[:, 1:-1, :-1]


def fHY(FHY, m, T, c):
    t = T + m * C.DT / 2
    z = np.sin(c * t) * (1 / c) * FHY
    return z[:, :-1, 1:-1]


def tf_trapz(y, axis=-2, dx=C.DX, rank=4):
    """"
    This is the trapz rule for tensors
    """
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    ret = tf.math.reduce_sum(dx * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)

    return ret


def tf_simp(y, axis=-2, dx=C.DX, rank=4):
    """"
    This is simpson rule for tensors
    """
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
    """
    this function recieves Hx or Hy and complete its values in the missing point according the chosen kernels.
    """
    rowup = tf.nn.conv2d(H, kernelup, strides=1, padding='VALID')
    rowdown = tf.nn.conv2d(H, kerneldown, strides=1, padding='VALID')
    a = tf.concat([rowup, H, rowdown], axis=1)
    colleft = tf.nn.conv2d(a, kernelleft, strides=1, padding='VALID')
    colright = tf.nn.conv2d(a, kernelright, strides=1, padding='VALID')
    return tf.concat([colleft, a, colright], axis=2)


def tf_diff(y, axis, rank=4):
    """"
    this is the same a np.diff, but for tensors
    """
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    ret = y[tuple(slice1)] - y[tuple(slice2)]
    return ret


def tf_simp3(y, axis=-2, dx=C.DX, rank=4):
    """"
    simpson rule of higher order
    """

    assert y.shape[axis] % 2 != 0
    nd = rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice3 = [slice(None)] * nd
    slice1[axis] = slice(2, None, 2)
    slice2[axis] = slice(1, -1, 2)
    slice3[axis] = slice(None, -2, 2)
    return tf.math.reduce_sum(dx * (y[tuple(slice1)] + 4 * y[tuple(slice2)] + y[tuple(slice3)]) / 3.0, axis=axis)


def tf_simp4(y, axis=-2, dx=C.DX, rank=4):
    """"
    simpson rule of higher order
    """
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


def amper(E, Hx, Hy, beta, delta, gamma, cfl=C.CFL):
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


def faraday(E, Hx, Hy, beta, delta, gamma, cfl=Constants.CFL):
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


# return tf.cast(tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), tf.cast(kernel,tf.dtypes.float64), strides=1, padding='VALID'), C.DTYPE)


def Dx(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


# return tf.cast(tf.nn.conv2d(tf.cast(B,tf.dtypes.float64), tf.cast(kernel,tf.dtypes.float64), strides=1, padding='VALID'), C.DTYPE)


def f_a(c, n, k1, k2):
    e = np.cos(c * n * C.DT) * (
            np.sin(C.PI * k1 * C.X) * np.sin(C.PI * k2 * C.Y) +
            np.sin(C.PI * k2 * C.X) * np.sin(
        C.PI * k1 * C.Y))

    hx = (1 / c) * np.sin(c * (C.DT / 2) * (2 * n + 1)) * (
            -C.PI * k2 * np.sin(C.PI * k1 * C.X) * np.cos(
        C.PI * k2 * (C.Y + C.DX / 2)) - C.PI * k1 * np.sin(
        C.PI * k2 * C.X) * np.cos(C.PI * k1 * (C.Y + C.DX / 2)))

    hy = (1 / c) * np.sin(c * (C.DT / 2) * (2 * n + 1)) * (
            C.PI * k1 * np.cos(C.PI * k1 * (C.X + C.DX / 2)) * np.sin(
        C.PI * k2 * C.Y) + C.PI * k2 * np.cos(
        C.PI * k2 * (C.X + C.DX / 2)) * np.sin(C.PI * k1 * C.Y))

    if k1 == k2:
        energy = 1
    else:
        energy = 1 / 2

    return e, hx[1:-1, :-1], hy[:-1, 1:-1], energy


def pad_function(input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])





# def loss_yee3(name, beta, delta, gamma, test_data):
#     """
#     another version of the function loss_yee
#     """
#
#     E1 = np.expand_dims(test_data['e'][0], axis=(0, -1))
#     Hx1 = np.expand_dims(test_data['hx'][0], axis=(0, -1))
#     Hy1 = np.expand_dims(test_data['hy'][0], axis=(0, -1))
#
#     le = []
#     lh = []
#     energy = []
#     divergence = []
#     for n in range(Constants.TIME_STEPS - 3):
#         E1 = amper(E1, Hx1, Hy1, beta, delta, gamma)
#         Hx1, Hy1 = faraday(E1, Hx1, Hy1, beta, delta, gamma)
#
#         # E1[0,1:Constants.N - 1, 1:Constants.N - 1,0] +=(Constants.DT / Constants.DX) * np.diff(Hy1[0,:,:,0], axis=0) - (Constants.DT / Constants.DX) * np.diff(Hx1[0,:,:,0], axis=1)
#         # Hx1[0,:,:,0] -= (Constants.DT / (Constants.DX)) * (np.diff(E1[0,1:-1, :,0], axis=1))
#         # Hy1[0,:,:,0] += (Constants.DT / (Constants.DX)) * (np.diff(E1[0,:, 1:-1,0], axis=0))
#
#         lh.append(relative_norm(Hx1[0, :, :, 0], test_data['hx'][n + 1]) +
#                   relative_norm(Hy1[0, :, :, 0], test_data['hy'][n + 1]))
#         le.append(relative_norm(E1[0, :, :, 0], test_data['e'][n + 1]))
#
#         # inte = tf_simp(tf_simp(E1 ** 2, rank=4), rank=3)
#         # inthx = tf_simp(tf_simp(Hx1 ** 2, rank=4), rank=3)
#         # inthy = tf_simp(tf_simp(Hy1 ** 2, rank=4), rank=3)
#         inte = tf_trapz(tf_trapz(E1 ** 2, rank=4), rank=3)
#         inthx = tf_trapz(tf_trapz(Hx1 ** 2, rank=4), rank=3)
#         inthy = tf_trapz(tf_trapz(Hy1 ** 2, rank=4), rank=3)
#
#         energy.append(tf.squeeze(inte + inthx + inthy))
#         y1 = tf.math.multiply(beta, Dy(Hy1, Constants.FILTER_BETA))
#         y2 = tf.math.multiply(delta, Dy(Hy1, Constants.FILTER_DELTA))
#         y3 = Dy(Hy1, Constants.FILTER_YEE)
#         dhydy = y1 + y2 + y3
#
#         x1 = tf.math.multiply(beta, Dx(Hx1, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
#         x2 = tf.math.multiply(delta, Dx(Hx1, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
#         x3 = Dx(Hx1, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))
#         dhxdx = x1 + x2 + x3
#
#         div = (dhydy[:, 1:-1, :, :] + dhxdx[:, :, 1:-1, :]) / Constants.DX
#         divergence.append(tf.reduce_max(abs(div)))
#         # ld.append( tf.reduce_max(abs(tf.squeeze(tf_diff(Hy1[:, 1:-1, :, :], axis=2) + tf_diff(Hx1[:, :, 1:-1, :], axis=1)))))
#
#     return divergence, energy, le, lh


def custom_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # return tf.math.reduce_mean(abs(y_true[:, 5:-5, 3:-3, :] - y_pred[:, 3:-3, 3:-3, :])) / C.CFL
    return tf.math.reduce_mean(abs(y_true - y_pred)) / C.CFL
    # return tf.math.reduce_mean(abs(y_true[:,5:-5,5:-5,] - y_pred[:,5:-5,5:-5,:])) / C.DT


def custom_loss3(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # return tf.math.reduce_mean(abs(y_true[:, 5:-5, 3:-3, :] - y_pred[:, 3:-3, 3:-3, :])) / C.CFL
    return tf.math.reduce_mean(tf.math.sin(abs(y_true - y_pred))) / C.CFL
    # return tf.math.reduce_mean(abs(y_true[7:-7, 7:-7] - y_pred[7:-7, 7:-7]))


def custom_loss_drp(y_true, y_pred):
    return abs(y_pred)


class DRP_LAYER(keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.pars1 = tf.Variable(0., trainable=False, dtype=C.DTYPE, name='beta')
        self.pars2 = tf.Variable(0., trainable=True, dtype=C.DTYPE, name='delta')
        self.pars3 = tf.Variable(0., trainable=False, dtype=C.DTYPE, name='gamma')

    def call(self, input):
        E1, Hx1, Hy1, E2, Hx2, Hy2, E3, Hx3, Hy3 = input

        # E_2 = amper(E1, Hx1, Hy1, self.pars1, (16*self.pars1-1)/24, -self.pars1/3)
        # Hx_2, Hy_2 = faraday(E2, Hx1, Hy1, self.pars1, (16*self.pars1-1)/24, -self.pars1/3)
        #
        # E_3 = amper(E_2, Hx_2, Hy_2, self.pars1, (16*self.pars1-1)/24, -self.pars1/3)
        # Hx_3, Hy_3 = faraday(E_3, Hx_2, Hy_2, self.pars1, (16*self.pars1-1)/24, -self.pars1/3)
        #
        # E_4 = amper(E_3, Hx_3, Hy_3, self.pars1, (16*self.pars1-1)/24, -self.pars1/3)
        # Hx_4, Hy_4 = faraday(E_4, Hx_3, Hy_3, self.pars1,(16*self.pars1-1)/24, -self.pars1/3)

        E_2 = amper(E1, Hx1, Hy1, self.pars1, self.pars2, self.pars3)
        Hx_2, Hy_2 = faraday(E_2, Hx1, Hy1, self.pars1, self.pars2, self.pars3)

        E_3 = amper(E_2, Hx_2, Hy_2, self.pars1, self.pars2, self.pars3)
        Hx_3, Hy_3 = faraday(E_3, Hx_2, Hy_2, self.pars1, self.pars2, self.pars3)

        E_4 = amper(E_3, Hx_3, Hy_3, self.pars1, self.pars2, self.pars3)
        Hx_4, Hy_4 = faraday(E_4, Hx_3, Hy_3, self.pars1, self.pars2, self.pars3)

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

# if __name__ == "__main__":
#     print(drp_loss(-1/24))
