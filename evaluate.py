import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from constants import Constants
from utils import *
from data_generator import *



# generate_data(Constants.K1_TRAIN, Constants.K2_TRAIN)
#generate_train_data()


create_test_data()


path = Constants.PATH

model1 = keras.models.load_model(path + 'checkpoint/mymodel_1net_nodiv.pkl', custom_objects={'custom_loss': custom_loss,'custom_loss3': custom_loss3 })
model1.load_weights( path+ 'checkpoint/model_weights_1net_nodiv.pkl').expect_partial()

#model2 = keras.models.load_model(path + 'mymodel_no_div.pkl', custom_objects={'custom_loss': custom_loss,'custom_loss3': custom_loss3 })
#model2.load_weights( path+ 'model_weights_no_div.pkl').expect_partial()

#with open(path + 'multiple_history.pkl', 'rb') as file:
#    history = pickle.load(file)
with open(path + 'ex_test.pkl', 'rb') as file:
    e_true = pickle.load(file)
with open(path + 'hx_x_test.pkl', 'rb') as file:
    hx_true =pickle.load(file)
with open(path + 'hy_x_test.pkl', 'rb') as file:
    hy_true = pickle.load(file)

l_yee = []
l_model = []
l_fourth = []
l_drp=[]
for i in range(Constants.TEST_NUM):
    E1 = tf.identity(tf.reshape(e_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N, Constants.N, 1]))
    Hx1 = tf.identity(tf.reshape(hx_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N - 2, Constants.N - 1, 1]))
    Hy1 = tf.identity(tf.reshape(hy_true[i * Constants.TIME_STEPS, :, :, :], [1, Constants.N - 1, Constants.N - 2, 1]))
    l_model.append(loss_model(model1, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    #l_yee.append(loss_model(model2, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))

    l_yee.append(loss_yee('Yee',0., 0., E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    l_fourth.append(loss_yee('4order',0., -1/24, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))
    l_drp.append(loss_yee('DRP', -0.125, -0.125, E1, Hx1, Hy1, e_true, hx_true, hy_true, i))


pickle.dump(l_yee, open(path+"l_yee.pkl", "wb"))
pickle.dump(l_fourth, open(path+"l_fourth.pkl", "wb"))
pickle.dump(l_model, open(path+"l_model.pkl", "wb"))
pickle.dump(l_drp, open(path+"l_drp.pkl", "wb"))

plt.plot(l_yee, "-b", label="Yee")
plt.plot(l_fourth, "-r", label="4th")
plt.plot(l_model, "-g", label="DL2", linestyle='dashed')
plt.plot(l_drp, "-d", label="DRP")
plt.legend(loc="upper left")
plt.show()
