from data_generator import create_test_data

import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import  amper, faraday, Dx, pad_function

from constants import Constants
path = Constants.PATH


from data_generator import create_test_data

create_test_data()


with open(path+'ex_test.pkl', 'rb') as file:
    e_true = tf.cast(pickle.load(file),tf.dtypes.float64)
with open(path+'hx_x_test.pkl', 'rb') as file:
    hx_true = tf.cast( pickle.load(file),tf.dtypes.float64)
with open(path+'hy_x_test.pkl', 'rb') as file:
    hy_true = tf.cast(pickle.load(file),tf.dtypes.float64)




# e_true=tf.cast(np.random.randint(5, size=(400, 100,100,1)),tf.dtypes.float64)
# hx_true=tf.cast(np.random.randint(5, size=(400, 98,99,1)),tf.dtypes.float64)
# hy_true=tf.cast(np.random.randint(5, size=(400, 99,98,1)),tf.dtypes.float64)

E1=tf.identity(tf.cast(tf.reshape(e_true[0,:,:,:],[1,Constants.N,Constants.N,1]),tf.dtypes.float64))
Hx1=tf.identity(tf.cast(tf.reshape(hx_true[0,:,:,:],[1,Constants.N-2,Constants.N-1,1]),tf.dtypes.float64))
Hy1=tf.identity(tf.cast(tf.reshape(hy_true[0,:,:,:],[1,Constants.N-1,Constants.N-2,1]),tf.dtypes.float64))

#E = np.squeeze(e_true[0,:,:,:]).copy()
#Hx = np.squeeze(hx_true[0,:,:,:]).copy()
#Hy = np.squeeze(hy_true[0,:,:,:]).copy()
#
# E = tf.cast(np.squeeze(e_true[0,:,:,:]).copy(),dtype=tf.dtypes.float32)
# Hx = tf.dtypes.float32(np.squeeze(hx_true[0,:,:,:]).copy())
# Hy = tf.dtypes.float32(np.squeeze(hy_true[0,:,:,:]).copy())
Z = 1
error=0
l=0
beta=0.
delta=0.
e_err=[]
hx_err=[]
hy_err=[]
for n in range(Constants.TIME_STEPS-1):
    #print(tf.reduce_mean(abs(np.array(E1[0, :, :, 0]) - E)))
    #E[1:Constants.N - 1, 1:Constants.N - 1] +=(Constants.DT / Constants.DX) * np.diff(Hy, axis=0) - (Constants.DT / Constants.DX) * np.diff(Hx, axis=1)
    #Hx -= (Constants.DT / (Constants.DX)) * (np.diff(E[1:-1, :], axis=1))
    #Hy += (Constants.DT / (Constants.DX)) * (np.diff(E[:, 1:-1], axis=0))
    E1=amper(tf.identity(E1),tf.identity(Hx1),tf.identity(Hy1), beta, delta)
    Hx1, Hy1= faraday(tf.identity(E1), tf.identity(Hx1), tf.identity(Hy1), beta, delta)
    e_err.append(tf.reduce_max(abs(E1[0, :, :, 0] - e_true[n + 1, :, :, 0])))
    hx_err.append(tf.reduce_max(abs(Hx1[0,:,:,0] - hx_true[n + 1, :, :, 0])))
    hy_err.append(tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[n + 1, :, :, 0])))

    # = (np.diff(Hx1[0,:,:,0], axis=0)[:, 1:-1] + np.diff(Hy1[0,:,:,0], axis=1)[1:-1, :]) / (2 * Constants.DX)
    #print(tf.reduce_max(abs(np.array(E1[0, :, :, 0]) - E[:,:])))
    #print('{:.20}'.format(tf.reduce_max(abs(Hx1[0, :, :, 0] - Hx[:, :]))))
    #S= tf.math.reduce_max(abs(np.array(Hx[:,:]) - hx_true[n + 1, :, :, 0]))
    #l += tf.reduce_mean(abs(E1[0,:,:,0] - e_true[n + 1, :, :, 0]))
#print(l/(Constants.TIME_STEPS-1))

plt.plot(e_err)
plt.plot(hx_err)
plt.plot(hy_err)
plt.show()



