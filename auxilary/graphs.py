import pickle
import tracemalloc

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from DRP_multiple_networks.constants import Constants

from DRP_multiple_networks.utils import custom_loss, custom_loss3, loss_yee2, loss_yee3, loss_yee, fd_solver, loss_yee4
from DRP_multiple_networks.data_generator import create_test_data
from DRP_multiple_networks.drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2

path = Constants.PATH

name = 'test_model1'
saving_path = path + 'Experiment_' + name + '_details/'
model1 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
model1.load_weights(
    saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()  # print(model1.trainable_weights)

l_model = []
l_fourth = []
l_drp = []

for N in [21,23]:
    time_steps=(N-1)*2+1
    h=1/(N-1)
    t_f=0.01
    dt=t_f/(time_steps-1)
    cfl=dt/h
    x1 = np.linspace(0., 1, N)
    x2 = np.linspace(0., 1, N)
    create_test_data(options='lt', h=h, dt=dt, t_f=t_f, time_steps=time_steps,
                     X1=x1, X2=x2)
    with open(path + 'test/test_data.pkl', 'rb') as file:
        test_data = pickle.load(file)

    data = {name: test_data[name][0] for name in list(test_data)}
    print(time_steps)

    for i in range(len(test_data['e'])):
        data = {name: test_data[name][i] for name in list(test_data)}
        loss_yee4('4order',0., -1 / 24, 0.,data, t_steps=time_steps, cfl=cfl)
        # l_model.append(
        #     loss_yee('model', model1.trainable_weights[0], model1.trainable_weights[1], model1.trainable_weights[2],
        #              data))
#
print(q)




for i in range(len(test_data['e'])):
    data = {name: test_data[name][i] for name in list(test_data)}

    l_fourth.append(loss_yee4('4order', 0., -1 / 24, 0., data))
    l_model.append(
        loss_yee4('model', model1.trainable_weights[0], model1.trainable_weights[1], model1.trainable_weights[2], data))
    # l_yee.append(loss_yee('Yee', 0, 0, data))
    # l_drp.append(loss_yee('DRP', 0., var, 0, data))
    # print(np.log(l_drp))

    print(l_model)
    # print(l_drp)
    print(l_fourth)
