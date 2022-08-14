import pickle
import tracemalloc

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import Constants
from utils import custom_loss, custom_loss3, loss_yee2, loss_yee3, loss_yee
from data_generator import create_test_data
from drp import calculate_DRP

path = Constants.PATH

create_test_data()
with open(path + 'test/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)



name = 'DL2i_N=41'
saving_path = path + 'Experiment_' + name + '_details/'
model1 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
model1.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()

name = 'DL3i_N=41'
saving_path = path + 'Experiment_' + name + '_details/'
model2 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})

model2.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()



l_yee = []
l_model = []
l_fourth = []
l_drp = []
l_model2=[]

print(model1.trainable_weights)
for i in range(len(test_data['e'])):
    data = {name: test_data[name][i] for name in list(test_data)}
    l_fourth.append(loss_yee('4order', 0., -1 / 24, data))
    l_model.append(loss_yee('model', 0, model1.trainable_weights[0], data))
    l_model2.append(loss_yee('model2', 0, model2.trainable_weights, data))
    l_yee.append(loss_yee('Yee', 0, 0, data))
    l_drp.append(loss_yee('DRP', 0., calculate_DRP(), data))
    print(l_drp)

    # lt_model=loss_yee2('model', 0, model1.trainable_weights[0], data)
    # lt_model2 = loss_yee2('model', 0, model2.trainable_weights[0], data)
    # lt_yee=loss_yee2('Yee', 0, 0, data)
    # lt_fourth=loss_yee2('4order', 0., -1/24, data)
    # #lt_fourth2 = loss_yee2('4order', 0., -0.04468, data)
    # lt_drp=loss_yee2('DRP', 0., calculate_DRP(), data)


# pickle.dump(l_yee, open(path+"figures/loss_yee.pkl", "wb"))
# pickle.dump(l_fourth, open(path+"figures/loss_fourth.pkl", "wb"))
# pickle.dump(l_model, open(path+"figures/loss_model1.pkl", "wb"))
# #pickle.dump(l_model2, open(path+"figures/loss_model2.pkl", "wb"))
# pickle.dump(l_drp, open(path+"figures/loss_drp.pkl", "wb"))
#
# pickle.dump(lt_yee, open(path+"figures/loss_time_yee.pkl", "wb"))
# pickle.dump(lt_fourth, open(path+"figures/loss_time_fourth.pkl", "wb"))
# pickle.dump(lt_model, open(path+"figures/loss_time_model1.pkl", "wb"))
# #pickle.dump(lt_model2, open(path+"figures/loss_time_model2.pkl", "wb"))
# pickle.dump(lt_drp, open(path+"figures/loss_time_drp.pkl", "wb"))

plt.plot(l_drp, '-r', label="drp")
plt.plot(l_model2, 'b', label='model2')
plt.plot(l_model, 'g', label='model1')
#plt.plot(lt_fourth2, 'r', label='other model')
# plt.plot(lt_fourth2, 'b', label='H error')
# plt.plot(np.array(lt_fourthe)*20, 'g', label='divergence')
# # plt.plot([(l_model[k]/l_model2[k]) for k in range(len(l_model))], "b", label="gpa")
plt.legend(loc="upper left")

plt.title('Error over time')
plt.show()
# print(1-3*model1.trainable_weights[0])
# print(model2.trainable_weights[0])
# print(1-3*calculate_DRP())
print(calculate_DRP())