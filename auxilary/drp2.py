import math

import numpy as np
import tensorflow as tf
import scipy.optimize as scop
from scipy.integrate import simps
from tensorflow import keras

from DRP_multiple_networks.constants import model_constants

"""
This file calculate the optimal coefficient for the drp loss (continuous version) from pi/2 to pi
"""

path = model_constants.PATH


def func(a, *args):
    Constants=model_constants

    X = np.linspace(math.pi/2, math.pi, model_constants.N*50)
    x, y = np.meshgrid(X, X, indexing='ij')
    y=x

    f = ((1 - a) ** 2) * (np.cos(3 * x) + np.cos(3 * y)) + \
        (6 * a - 6 * a ** 2) * (np.cos(2 * x) + np.cos(2 * y)) + \
        (15 * a ** 2 - 6 * a) * (np.cos(x) + np.cos(y))
    omega_over_k = (2 / (Constants.CFL * np.sqrt(x ** 2 + y ** 2))) * np.arcsin(
        Constants.CFL * np.sqrt((1 / 18) * (20 * a ** 2 - 4 * a + 2 - f)))
    z = abs((omega_over_k - 1)**2)

    return simps([simps(zz_x, X) for zz_x in z], X)




def calculate_DRP2():
    Constants=model_constants
    opt = 100
    x = 9/8
    for i in range(4):
        init = np.random.rand(1)

        res = scop.minimize(func, init, args=(Constants), method='SLSQP', bounds=scop.Bounds(0, 20),
                            options=dict(disp=False, iprint=2, ftol=1e-18))
        if res['fun'] < opt:
            x = res['x']
            opt = res['fun']

    return (1 - x) / 3



def func_extended( pars, *args):
    par1, par2, par3 = pars


    Constants = model_constants

    X = np.linspace(math.pi/2, 2*math.pi, Constants.N*50)
    x, y = np.meshgrid(X, X, indexing='ij')

    f=2*np.cos(y/2)*(par1*np.sin(x/2)+par3*(np.sin(3*x/2)))+par2*np.sin(3*x/2)+(1-2*par1-3*par2-6*par3)*np.sin(x/2)
    g= 2*np.cos(x/2)*(par1*np.sin(y/2)+par3*(np.sin(3*y/2)))+par2*np.sin(3*y/2)+(1-2*par1-3*par2-6*par3)*np.sin(y/2)
    omega_over_k=(2 / (Constants.CFL * np.sqrt(x ** 2 + y ** 2)))*np.arcsin(Constants.CFL*np.sqrt(f**2+g**2))
    z=abs((omega_over_k - 1)**2)

    return simps([simps(zz_x, X) for zz_x in z], X)



def calculate_DRP2_extended():
    Constants=model_constants
    opt = 100
    x = (0,0,0)
    for i in range(1):
        init = x

        res = scop.minimize(func_extended, init, args=(Constants), method='SLSQP', bounds=scop.Bounds(-0.5, 5),
                            options=dict(disp=False, iprint=2, ftol=1e-8))
        print(res['fun'])
        if res['fun'] < opt:
            x = res['x']
            opt = res['fun']


    return x

if __name__ == "__main__":
   x1=calculate_DRP2()
   print(x1)
