

import numpy as np
import pickle
import tensorflow as tf

from utils import  amper, faraday, Dx, pad_function

from constants import Constants
with open('/Users/idanversano/documents/pycharm/files/ex_test.pkl', 'rb') as file:
    e_true = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('/Users/idanversano/documents/pycharm/files/hx_x_test.pkl', 'rb') as file:
    hx_true = tf.cast( pickle.load(file),tf.dtypes.float32)
with open('/Users/idanversano/documents/pycharm/files/hy_x_test.pkl', 'rb') as file:
    hy_true = tf.cast(pickle.load(file),tf.dtypes.float32)



# e_true=tf.cast(np.random.randint(5, size=(400, 100,100,1)),tf.dtypes.float64)
# hx_true=tf.cast(np.random.randint(5, size=(400, 98,99,1)),tf.dtypes.float64)
# hy_true=tf.cast(np.random.randint(5, size=(400, 99,98,1)),tf.dtypes.float64)

E1=tf.identity(tf.cast(tf.reshape(e_true[0,:,:,:],[1,Constants.N,Constants.N,1]),tf.dtypes.float64))
Hx1=tf.identity(tf.cast(tf.reshape(hx_true[0,:,:,:],[1,Constants.N-2,Constants.N-1,1]),tf.dtypes.float64))
Hy1=tf.identity(tf.cast(tf.reshape(hy_true[0,:,:,:],[1,Constants.N-1,Constants.N-2,1]),tf.dtypes.float64))

E = np.squeeze(e_true[0,:,:,:]).copy()
Hx = np.squeeze(hx_true[0,:,:,:]).copy()
Hy = np.squeeze(hy_true[0,:,:,:]).copy()
#
# E = tf.cast(np.squeeze(e_true[0,:,:,:]).copy(),dtype=tf.dtypes.float32)
# Hx = tf.dtypes.float32(np.squeeze(hx_true[0,:,:,:]).copy())
# Hy = tf.dtypes.float32(np.squeeze(hy_true[0,:,:,:]).copy())
Z = 1
error=0
l=0
w = 1.

for n in range(Constants.TIME_STEPS-1):
    #print(tf.reduce_mean(abs(np.array(E1[0, :, :, 0]) - E)))
    E[1:Constants.N - 1, 1:Constants.N - 1] +=(Constants.DT / Constants.DX) * np.diff(Hy, axis=0) - (Constants.DT / Constants.DX) * np.diff(Hx, axis=1)
    Hx -= (Constants.DT / (Constants.DX)) * (np.diff(E[1:-1, :], axis=1))
    Hy += (Constants.DT / (Constants.DX)) * (np.diff(E[:, 1:-1], axis=0))


    E1=amper(tf.identity(E1),tf.identity(Hx1),tf.identity(Hy1),w)

    Hx1, Hy1= faraday(tf.identity(E1), tf.identity(Hx1), tf.identity(Hy1),w)
    print(Hx1.shape)
    deriv = (np.diff(Hx1[0,:,:,0], axis=0)[:, 1:-1] + np.diff(Hy1[0,:,:,0], axis=1)[1:-1, :]) / (2 * Constants.DX)
    print(deriv)
    #print(tf.reduce_max(abs(np.array(E1[0, :, :, 0]) - E[:,:])))
    #print('{:.20}'.format(tf.reduce_max(abs(Hx1[0, :, :, 0] - Hx[:, :]))))
    #S= tf.math.reduce_max(abs(np.array(Hx[:,:]) - hx_true[n + 1, :, :, 0]))
    #print(S)
    l += tf.reduce_mean(abs(E - e_true[n + 1, :, :, 0]))
print(l/(Constants.TIME_STEPS-1))





