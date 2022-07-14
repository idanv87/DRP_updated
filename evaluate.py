import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from constants import Constants
from utils import loss_yee, loss_model, custom_loss, Dx
from data_generator import create_test_data

create_test_data()

path = Constants.PATH

model = keras.models.load_model(path + 'mymodel_multiple', custom_objects={'custom_loss': custom_loss})
with open(path + 'multiple_history.pkl', 'rb') as file:
    history = pickle.load(file)
with open(path + 'ex_test.pkl', 'rb') as file:
    e_true = tf.cast(pickle.load(file), tf.dtypes.float64)
with open(path + 'hx_x_test.pkl', 'rb') as file:
    hx_true = tf.cast(pickle.load(file), tf.dtypes.float64)
with open(path + 'hy_x_test.pkl', 'rb') as file:
    hy_true = tf.cast(pickle.load(file), tf.dtypes.float64)

l_yee = []
l_model = []
l_fourth = []
l_drp=[]
for i in range(len(Constants.K1_TEST) * len(Constants.K2_TEST)):
    E1 = tf.identity(tf.reshape(e_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N, Constants.N, 1]))
    Hx1 = tf.identity(tf.reshape(hx_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N - 2, Constants.N - 1, 1]))
    Hy1 = tf.identity(tf.reshape(hy_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N - 1, Constants.N - 2, 1]))
    l_yee.append(loss_yee('Yee',0., 0., E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    l_fourth.append(loss_yee('4order',0., -1/24, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    l_drp.append(loss_yee('DRP', 0., -1 / 24, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    # make that dt =0.1 dx
    l_model.append(loss_model(model, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
pickle.dump(l_yee, open(path+"l_yee.pkl", "wb"))
pickle.dump(l_yee, open(path+"l_fourth.pkl", "wb"))
pickle.dump(l_yee, open(path+"l_model.pkl", "wb"))
pickle.dump(l_yee, open(path+"l_drp.pkl", "wb"))

plt.plot(l_yee)
plt.plot(l_fourth)
plt.plot(l_model)
plt.plot(l_drp)
plt.show()
