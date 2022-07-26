import numpy as np
import tensorflow as tf
import scipy.optimize as scop

from  constants import *


def func(a, *args):
    h=Constants.DX
    k=np.sqrt(k1**2+k2**2)

    f=((1-a)**2)*(np.cos(3*k1*h)+np.cos(3*k2*h)) + \
      (6*a-6*a**2 )*(np.cos(2*k1*h)+np.cos(2*k2*h)) + \
      (15*a**2-6*a)*(np.cos(k1*h)+np.cos(k2*h))
    omega = (2/Constants.DT)*np.arcsin((Constants.DT/h)*np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))

    return tf.math.reduce_max(abs(omega-k))
def cons1(a, *args):
    return 1-(1/18)*((Constants.DT/Constants.DX)**2)*np.maximum(8*((a+0.5)**2), 64*((a-0.25)**2))
def cons2(a, *args):
    return (1/18)*((Constants.DT/Constants.DX)**2)*np.maximum(8*((a+0.5)**2), 64*((a-0.25)**2))

k1, k2 = np.meshgrid(Constants.PI*np.arange(1,20), Constants.PI*np.arange(1,20), indexing='ij')

cons=[{'type':'ineq', 'fun':cons1 }, {'type':'ineq', 'fun':cons2 }]
a=np.array([0.1])


res = scop.minimize(func, a, args=(k1, k2), method='SLSQP',bounds= scop.Bounds(0,10),
                     constraints=cons,
                     options=dict(disp=False, iprint=2))
print(cons1(10))










