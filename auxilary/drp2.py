import math

import numpy as np
import tensorflow as tf
import scipy.optimize as scop
from scipy.integrate import simps
from tensorflow import keras

from DRP_multiple_networks.constants import Constants

"""
This file calculate the optimal coefficient for the drp loss (continuous version) from pi/2 to pi
"""

path = Constants.PATH


def func(a, *args):
    X = np.linspace(math.pi / 2, math.pi, 800)
    x, y = np.meshgrid(X, X, indexing='ij')

    f = ((1 - a) ** 2) * (np.cos(3 * x) + np.cos(3 * y)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * x) + np.cos(2 * y)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(x) + np.cos(y))
    omega_over_k = (2 / (Constants.CFL * np.sqrt(x ** 2 + y ** 2))) * np.arcsin(
        Constants.CFL * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))
    z = abs(omega_over_k - 1)

    return simps([simps(zz_x, X) for zz_x in z], X)

def cons1(a, *args):
    return 1 - (1 / 18) * (Constants.CFL ** 2) * np.maximum(8 * ((a + 0.5) ** 2), 64 * ((a - 0.25) ** 2))


def cons2(a, *args):
    return (1 / 18) * (Constants.CFL ** 2) * np.maximum(8 * ((a + 0.5) ** 2), 64 * ((a - 0.25) ** 2))

cons = [{'type': 'ineq', 'fun': cons1}, {'type': 'ineq', 'fun': cons2}]
def calculate_DRP2():
    opt = 100
    x = -1/24
    for i in range(4):
        init = np.random.rand(1)

        res = scop.minimize(func, init, args=(), method='SLSQP', bounds=scop.Bounds(0, 20),
                            constraints=cons,
                            options=dict(disp=False, iprint=2, ftol=1e-18))
        if res['fun'] < opt:
            x = res['x']
            opt = res['fun']

    print(cons1(x))
    print(x)
    return (1 - x) / 3



