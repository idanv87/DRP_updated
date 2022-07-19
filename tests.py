import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from constants import Constants
from utils import *
from data_generator import create_test_data

F=tf.cast(np.exp(Constants.Y).reshape(1,Constants.N,Constants.N,1),tf.dtypes.float32)
#dFy_diff=tf_diff(F,axis=2,rank=4)

dFy_conv = Dy(F, Constants.FILTER_YEE)

#print(tf.math.reduce_max(abs(dFy_diff[:,1:-1,1:-1,:]-dFy_conv)))



#dFy3=np.diff(F,axis=2)



#print(tf.math.reduce_max(abs(dFy-dFy3)))


# x1 = tf.math.multiply(beta, Dx(E, tf.transpose(Constants.FILTER_BETA, perm=[1, 0, 2, 3])))
# x2 = tf.math.multiply(delta, Dx(E, tf.transpose(Constants.FILTER_DELTA, perm=[1, 0, 2, 3])))
# x3 = Dx(E, tf.transpose(Constants.FILTER_YEE, perm=[1, 0, 2, 3]))

#dFx = tf.pad(x1 + x2 + x3, pad3) + \
#     tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])), Constants.PADEY_FORWARD)[:,
#     :, 1:-1, :] + \
#     tf.pad(Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADEY_BACKWARD)[
#     :, :, 1:-1, :]
