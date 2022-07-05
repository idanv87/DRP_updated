

import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import matplotlib.pyplot as plt
import torch
import math
import matplotlib.animation as animation

from constants import Constants
with open('/Users/idanversano/documents/pycharm/files/e_test.pkl', 'rb') as file:
    e_true = pickle.load(file)
with open('/Users/idanversano/documents/pycharm/files/hx_test.pkl', 'rb') as file:
    hx_true = pickle.load(file)
with open('/Users/idanversano/documents/pycharm/files/hy_test.pkl', 'rb') as file:
    hy_true = pickle.load(file)

with open('/Users/idanversano/documents/pycharm/files/ex.pkl', 'rb') as file:
    ex = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('/Users/idanversano/documents/pycharm/files/ey.pkl', 'rb') as file:
    ey = tf.cast(pickle.load(file),tf.dtypes.float32)
# Grid parameters.
k1, k2 = Constants.K1_TEST[0], Constants.K2_TEST[0]




E = np.squeeze(e_true[0,:,:,:]).copy()
Hx = np.squeeze(hx_true[0,:,:,:]).copy()
Hy = np.squeeze(hy_true[0,:,:,:]).copy()


Z = 1
error=0
l_yee=0
for n in range(Constants.TIME_STEPS):
    E[1:Constants.N - 1, 1:Constants.N - 1] = E[1:Constants.N - 1, 1:Constants.N - 1] + \
        (Constants.DT / Constants.DX) * np.diff(Hy,axis=0) - (Constants.DT / Constants.DY) * np.diff(Hx,axis=1)
    Hx -=  (Constants.DT / ( Constants.DY)) * (np.diff(E[1:-1,:],axis=1))
    Hy +=  (Constants.DT / ( Constants.DX)) * (np.diff(E[:,1:-1],axis=0))
    l_yee+=tf.reduce_mean(abs(E-ey[n,0:Constants.N,:,0]))

print(l_yee/(Constants.TIME_STEPS))
