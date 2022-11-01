import pickle
import tracemalloc

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import Constants

from utils import custom_loss, custom_loss3, loss_yee
from data_generator import create_test_data
from drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2

path = Constants.PATH

create_test_data()

with open(path + 'test/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)

data = {name: test_data[name][0] for name in list(test_data)}

name = 'test_model1'
saving_path = path + 'Experiment_' + name + '_details/'
model1 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
model1.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()#print(model1.trainable_weights)



#
name = 'dl1'
saving_path = path + 'Experiment_' + name + '_details/'
model2 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})

model2.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()

name = 'dl4'
saving_path = path + 'Experiment_' + name + '_details/'
model3 = keras.models.load_model(saving_path + 'model.pkl',
                                 custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})

model3.load_weights(saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()



l_yee = []
l_model = []
l_fourth = []
l_drp = []
l_modeldl1 = []
l_modeldl4=[]

for i in range(len(test_data['e'])):
    data = {name: test_data[name][i] for name in list(test_data)}

    # aux = fd_solver(0., 0., data)
    # aux4 = fd_solver(0., -1 / 24, data)
    # aux_drp = fd_solver(0., calculate_DRP(), data)
    # plt.plot(aux)
    # plt.plot(aux4, 'r', label="4")
    # # plt.plot(aux_drp, 'g', label="drp")
    # plt.legend()
    # plt.show()
    # var=calculate_DRP2()
    var=-0.09165107

    l_fourth.append(loss_yee('4order', 0., -1 / 24, 0., data))
    l_yee.append(loss_yee('2order', 0., 0., 0., data))

    # l_modeldl1.append(loss_yee('dl1', 0., model2.trainable_weights[0],0., data))
    # l_model.append(loss_yee('dl3', model1.trainable_weights[0], model1.trainable_weights[1],model1.trainable_weights[2], data))
    # l_drp.append(loss_yee('DRP', 0., var, 0, data))
    # l_modeldl4.append(loss_yee('dl4', model3.trainable_weights[0],(16*model3.trainable_weights[0]-1)/24 , -model3.trainable_weights[0]/3, data))

    # print(np.log(l_drp))
    #
    #
    # l2_tot=[-19.50439923,-19.76743022,-19.999877]
    # #l_tot=np.array([-21.93620312,-23.78711441, -25.07347991, -26.05638995, -26.85038231, -27.51571427, -28.08792702, -28.58969093])
    # #dx_tot=np.log([1/20,1/30,1/40,1/50, 1/60, 1/70, 1/80, 1/90])
    # dx_tot=np.log([ 1/70, 1/80, 1/90])
    # print(np.divide(np.diff(l2_tot), np.diff(dx_tot)))
    # print(l_modeldl4)
    # print(l_modeldl1)
    # print(l_model)
    print(l_fourth)
    print(l_yee)
    # print(l_drp)

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
print(q)
plt.plot(l_drp, '-r', label="drp")
plt.plot(l_model1, 'b', label='model1')
plt.plot(l_fourth, 'g', label='fourth')
# plt.plot(lt_fourth2, 'r', label='other model')
# plt.plot(lt_fourth2, 'b', label='H error')
# plt.plot(np.array(lt_fourthe)*20, 'g', label='divergence')
# # plt.plot([(l_model[k]/l_model2[k]) for k in range(len(l_model))], "b", label="gpa")
plt.legend(loc="upper left")

plt.title('Error over time')
plt.show()
# print(1-3*model1.trainable_weights[0])
# print(model2.trainable_weights[0])
# print(1-3*calculate_DRP())
# print(calculate_DRP())
