
import numpy as np
import tensorflow as tf

from DRP_multiple_networks.constants import Constants

C = Constants()


def relative_norm(A, B, p=2):
    return tf.math.reduce_mean(abs(A-B))


def fE(t, x, y, k1, k2, c):
    return np.cos(c * t) * np.sin(C.PI * k1 * x) * np.sin(C.PI * k2 * y)


def fHX(t, x, y, k1, k2, c, h=C.DX):
    z = np.sin(c * t) * (1 / c) * (
            -C.PI * k2 * np.sin(C.PI * k1 * x) * np.cos(
        C.PI * k2 * (y + h / 2))
    )
    return z[:, 1:-1, :-1]


def fHY(t, x, y, k1, k2, c, h=C.DX):
    z = np.sin(c * t) * (1 / c) * (
            C.PI * k1 * np.cos(C.PI * k1 * (x + h / 2)) * np.sin(
        C.PI * k2 * y)
    )
    return z[:, :-1, 1:-1]


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


def loss_drp(h, dt, k1, k2, a):
        k = np.sqrt(k1 ** 2 + k2 ** 2)

        f = ((1 - a) ** 2) * (np.cos(3 * k1 * h) + np.cos(3 * k2 * h)) + \
            (6 * a - 6 * a ** 2) * (np.cos(2 * k1 * h) + np.cos(2 * k2 * h)) + \
            (15 * a ** 2 - 6 * a) * (np.cos(k1 * h) + np.cos(k2 * h))
        omega = (2 / dt) * np.arcsin((dt / h) * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

        return abs(omega / k - 1)