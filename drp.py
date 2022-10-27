import numpy as np
import tensorflow as tf
import scipy.optimize as scop
from tensorflow import keras


from DRP_multiple_networks.constants import Constants
from DRP_multiple_networks.utils import custom_loss, custom_loss3, loss_yee2, loss_yee3, loss_yee

"""
This file calculate the optimal coefficient for the drp loss (discrete version) over the selected modes
given in the file constants.k_drp
"""


path = Constants.PATH



def loss(a, *args):
    k1, k2 = np.meshgrid(Constants.PI * Constants.K1_DRP, Constants.PI * Constants.K2_DRP, indexing='ij')

    h = Constants.DX
    k = np.sqrt(k1 ** 2 + k2 ** 2)

    f = ((1 - a) ** 2) * (np.cos(3 * k1 * h) + np.cos(3 * k2 * h)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * k1 * h) + np.cos(2 * k2 * h)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(k1 * h) + np.cos(k2 * h))
    omega = (2 / Constants.DT) * np.arcsin(Constants.CFL * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    return tf.math.reduce_sum(abs(omega / k - 1))


def func(a, *args):
    k1, k2 = np.meshgrid(Constants.PI * Constants.K1_DRP, Constants.PI * Constants.K2_DRP, indexing='ij')

    h = Constants.DX
    k = np.sqrt(k1 ** 2 + k2 ** 2)

    f = ((1 - a) ** 2) * (np.cos(3 * k1 * h) + np.cos(3 * k2 * h)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * k1 * h) + np.cos(2 * k2 * h)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(k1 * h) + np.cos(k2 * h))
    omega = (2 / Constants.DT) * np.arcsin(Constants.CFL * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    return tf.math.reduce_mean(abs(omega / k - 1))
    # return tf.math.reduce_mean((omega/k-1)**2)


def cons1(a, *args):
    return 1 - (1 / 18) * (Constants.CFL ** 2) * np.maximum(8 * ((a + 0.5) ** 2), 64 * ((a - 0.25) ** 2))


def cons2(a, *args):
    return (1 / 18) * (Constants.CFL ** 2) * np.maximum(8 * ((a + 0.5) ** 2), 64 * ((a - 0.25) ** 2))


def cons3(a, *args):
    return 18 / (np.maximum(8 * ((a + 0.5) ** 2), 64 * ((a - 0.25) ** 2)))


def calculate_DRP():
    k1, k2 = np.meshgrid(Constants.PI * Constants.K1_DRP, Constants.PI * Constants.K2_DRP, indexing='ij')
    opt = 100
    x=0.
    cons = [{'type': 'ineq', 'fun': cons1}, {'type': 'ineq', 'fun': cons2}]
    for i in range(4):
        init = np.random.rand(1)

        res = scop.minimize(func, init, args=(k1, k2), method='SLSQP', bounds=scop.Bounds(0, 20),
                            constraints=cons,
                            options=dict(disp=False, iprint=2, ftol=1e-18))
        if res['fun']<opt:
            x = res['x']
            opt=res['fun']



    return (1 - x) / 3



