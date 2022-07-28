import pickle
import tracemalloc

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import Constants
from utils import custom_loss, custom_loss3, loss_model, loss_yee
from data_generator import create_test_data
from drp import calculate_DRP

path = Constants.PATH

create_test_data()
print('new test data created')
with open(path + 'test/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)

name = '1001_N=20'
saving_path = path + 'Experiment_' + name + '_details/'
model1 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})

model1.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()

with open(saving_path + 'experiment_' + name + '_details.pkl', 'rb') as file:
    model_parameters = pickle.load(file)

print("model trained with N=" + str(model_parameters["params"]["N"]))
print("model trained with CFL=" + str(model_parameters["params"]["CFL"]))

l_yee = []
l_model = []
l_fourth = []
l_drp = []

for i in range(len(test_data['ex'])):
    data = {name: test_data[name][i] for name in list(test_data)}

    l_model.append(loss_yee('model', 0, model1.trainable_weights[0], data))
    l_yee.append(loss_yee('Yee', 0, 0, data))
    l_fourth.append(loss_yee('4order', 0., -1 / 24, data))
    l_drp.append(loss_yee('DRP', 0., calculate_DRP(), data))

# pickle.dump(l_yee, open(path+"l_yee.pkl", "wb"))
# pickle.dump(l_fourth, open(path+"l_fourth.pkl", "wb"))
# pickle.dump(l_model, open(path+"l_model.pkl", "wb"))
# pickle.dump(l_drp, open(path+"l_drp.pkl", "wb"))

plt.plot(l_yee, "b", label="Yee")
plt.plot(l_fourth, "r", label="4th")
plt.plot(l_model, "g", label="DL2", linestyle='dashed')
plt.plot(l_drp, "-d", label="DRP")
plt.legend(loc="upper left")
plt.show()

