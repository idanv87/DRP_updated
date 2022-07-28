import numpy as np
import tensorflow as tf
import scipy.optimize as scop

from  constants import Constants

k1, k2 = np.meshgrid(Constants.K1_TRAIN, Constants.K2_TRAIN, indexing='ij')


def func(a, *args):
    h=Constants.DX
    k=np.sqrt(k1**2+k2**2)

    f=((1-a)**2)*(np.cos(3*k1*h)+np.cos(3*k2*h)) + \
      (6*a-6*a**2 )*(np.cos(2*k1*h)+np.cos(2*k2*h)) + \
      (15*a**2-6*a)*(np.cos(k1*h)+np.cos(k2*h))
    omega = (2/Constants.DT)*np.arcsin(Constants.CFL*np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    return tf.math.reduce_max(abs(omega-k))
def cons1(a, *args):
    return 1-(1/18)*(Constants.CFL**2)*np.maximum(8*((a+0.5)**2), 64*((a-0.25)**2))
def cons2(a, *args):
    return (1/18)*(Constants.CFL**2)*np.maximum(8*((a+0.5)**2), 64*((a-0.25)**2))

def calculate_DRP():
    init=np.random.rand(1)
    cons = [{'type': 'ineq', 'fun': cons1}, {'type': 'ineq', 'fun': cons2}]

    res = scop.minimize(func, init, args=(k1, k2), method='SLSQP', bounds=scop.Bounds(-2, 2),
                        constraints=cons,
                        options=dict(disp=False, iprint=2))

    return (1 - res['x']) / 3
















