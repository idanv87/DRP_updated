
import numpy as np
import tensorflow as tf

from DRP_multiple_networks.constants import Constants, model_constants




def relative_norm(A, B, p=2):

    return tf.math.reduce_mean((abs(A-B)))
    # return tf.math.reduce_mean((abs(tf.math.multiply(A-B,A-B))))


def fE(t, x, y, k1, k2, c):
    return np.cos(c * t) * np.sin(Constants.PI * k1 * x) * np.sin(Constants.PI * k2 * y)


def fHX(t, x, y, k1, k2, c, h):
    z = np.sin(c * t) * (1 / c) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * x) * np.cos(
        Constants.PI * k2 * (y + h / 2))
    )
    return z[:, 1:-1, :-1]


def fHY(t, x, y, k1, k2, c, h):
    z = np.sin(c * t) * (1 / c) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (x + h / 2)) * np.sin(
        Constants.PI * k2 * y)
    )
    return z[:, :-1, 1:-1]
#
# def dr_calculator_extended():
#     X = np.linspace(math.pi / 2, math.pi, model_constants.N * 50)
#     x, y = np.meshgrid(X, X, indexing='ij')
#     loss_drp_extended(model_constants,X,X, models['dl(2,3)'][0], models['dl(2,3)'][1], models['dl(2,3)'][2])
#
# #


def dim_red1(dic, m):
    d = {key: dic[key].copy() for key in dic.keys()}
    for key in ['e', 'hx', 'hy', 'energy']:
        for i in range(len(d[key])):
            if m != 3:
                d[key][i] = d[key][i][m:m - 3]
            else:
                d[key][i] = d[key][i][m:]

    return list([np.expand_dims(np.vstack(d[key]), axis=-1) for key in ['e', 'hx', 'hy', 'energy']])


def dim_red2(dic, m):
    d = {key: dic[key].copy() for key in dic.keys()}
    for key in ['e', 'hx', 'hy']:
        for i in range(len(d[key])):
            if m != 3:
                d[key][i] = d[key][i][m:m - 3]
            else:
                d[key][i] = d[key][i][m:]

    return list([np.expand_dims(np.vstack(d[key]), axis=-1) for key in ['e', 'hx', 'hy']])


def loss_drp(Constants, x, y, a):

    f = ((1 - a) ** 2) * (np.cos(3 * x) + np.cos(3 * y)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * x) + np.cos(2 * y)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(x) + np.cos(y))
    omega_over_k = (2 / (Constants.CFL * np.sqrt(x ** 2 + y ** 2))) * np.arcsin(
        Constants.CFL * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))
    z = abs(omega_over_k - 1)

    return z

def loss_drp_extended(Constants, x, y, par1, par2, par3):
    f=2*np.cos(y/2)*(par1*np.sin(x/2)+par3*(np.sin(3*x/2)))+par2*np.sin(3*x/2)+(1-2*par1-3*par2-6*par3)*np.sin(x/2)
    g= 2*np.cos(x/2)*(par1*np.sin(y/2)+par3*(np.sin(3*y/2)))+par2*np.sin(3*y/2)+(1-2*par1-3*par2-6*par3)*np.sin(y/2)
    omega_over_k=(2 / (Constants.CFL * np.sqrt(x ** 2 + y ** 2)))*np.arcsin(model_constants.CFL*np.sqrt(f**2+g**2))
    return abs(omega_over_k-1)



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


def tf_simp3(y, axis=-2, dx=0.1, rank=4):
    """"
    simpson rule of higher order for tensors
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


def tf_simp4(y, axis=-2, dx=0.1, rank=4):
    """"
    simpson rule of higher order for tesnors
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

def tf_trapz(y, axis=-2, dx=0.1, rank=4):
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


def tf_simp(y, axis=-2, dx=0.1, rank=4):
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

