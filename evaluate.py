import pickle

from tensorflow import keras
import tensorflow as tf
import numpy as np

from constants import Constants
from utils import amper, faraday

with open('/Users/idanversano/documents/pycharm/files/model_history.pkl', 'rb') as file:
    history = pickle.load(file)
with open('/Users/idanversano/documents/pycharm/files/ex_test.pkl', 'rb') as file:
    e_true = tf.cast(pickle.load(file),tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hx_x_test.pkl', 'rb') as file:
    hx_true = tf.cast( pickle.load(file),tf.dtypes.float64)
with open('/Users/idanversano/documents/pycharm/files/hy_x_test.pkl', 'rb') as file:
    hy_true = tf.cast(pickle.load(file),tf.dtypes.float64)

model = keras.models.load_model('/Users/idanversano/documents/pycharm/files/mymodel')

E1=tf.identity(tf.cast(tf.reshape(e_true[0,:,:,:],[1,Constants.N,Constants.N,1]),tf.dtypes.float64))
Hx1=tf.identity(tf.cast(tf.reshape(hx_true[0,:,:,:],[1,Constants.N-2,Constants.N-1,1]),tf.dtypes.float64))
Hy1=tf.identity(tf.cast(tf.reshape(hy_true[0,:,:,:],[1,Constants.N-1,Constants.N-2,1]),tf.dtypes.float64))

l_yee=0
w=model.trainable_weights
w=9/8
print(w)
for n in range(Constants.TIME_STEPS-1):
    E1 = amper(E1, Hx1, Hy1, w)
    Hx1, Hy1 = faraday(E1, Hx1, Hy1, w)
    l_yee += tf.reduce_max(abs(E1[0,:,:,0] - e_true[n + 1, :, :, 0]))+ \
             tf.reduce_max(abs(Hx1[0, :, :, 0] - hx_true[n + 1, :, :, 0]))+ \
             tf.reduce_max(abs(Hy1[0, :, :, 0] - hy_true[n + 1, :, :, 0]))
print(l_yee/(3*(Constants.TIME_STEPS-1)))







