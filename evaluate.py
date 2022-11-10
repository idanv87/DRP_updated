import numpy as np
import math
import pickle

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from pylab import *

from DRP_multiple_networks.drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2, func
from DRP_multiple_networks.constants import Constants, model_constants
from DRP_multiple_networks.data_generator import create_test_data
from DRP_multiple_networks.auxilary.aux_functions import loss_drp, loss_drp_extended
from utils import custom_loss, custom_loss3, loss_yee
from DRP_multiple_networks.auxilary.aux_functions import fE, fHX, fHY

path = Constants.PATH



# var=calculate_DRP2()
var=0

models = {'Yee(2,0)': [0,0.,0], 'Yee(4,0)': [0,-1 / 24,0], 'drp(2,1)': [0,var,0], 'dl(2,1)': [], 'dl(2,3)_all': [], 'dl(2,3)': [],
          'dl(4,1)': [], 'model_test': []}

for name in ['dl(2,1)', 'dl(2,3)', 'dl(4,1)', 'dl(2,3)_all', 'model_test']:
    saving_path = path + 'Experiment_' + name + '_details/'
    model = keras.models.load_model(saving_path + 'model.pkl',
                                    custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
    model.load_weights(
        saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()
    if name=='dl(2,1)':
        # models[name]=[model.trainable_weights[0], (16 * model.trainable_weights[0] - 1) / 24,
                         # -model.trainable_weights[0] / 3]
        models[name]=[0., model.trainable_weights[0],0]

    else:
       models[name] = model.trainable_weights




def dr_calculator(names, save=(False,'False')):
    X = np.linspace(0.5 * math.pi, math.pi, model_constants.N * 50)
    x, y = np.meshgrid(X, X, indexing='ij')
    fig, ax1 = plt.subplots(1, sharex=False, sharey=False)

    for name in names:
        assert name in list(models)
        ax1.plot(X / math.pi, loss_drp_extended(model_constants, X, X, models[name][0], models[name][1],
                                                    models[name][2]), label=name)

    plt.legend(loc="upper left")
    plt.xlabel(r'$\frac{k}{\pi}$')
    plt.ylabel('dispersion error')


    if save[0]:
        print(' saved as:dispersion_figure' + str(save[1]) + '.eps')
        plt.savefig(Constants.FIGURES_PATH + 'dispersion_figure'+save[1]+'.eps', format='eps',
                    bbox_inches='tight')
        plt.show()
    return 1


def error_print(names, n=None, x=None, t=None, time_steps=None, k1_test=None, k2_test=None, solve=True, save=('False','False')):
    '''
    This function recieve model names to evaluate over
    test data
    '''

    if solve:
        solve_equation(names, n, x, t, time_steps, k1_test, k2_test)

    with open(path + "figures/losses.pkl", 'rb') as file:
        loss = pickle.load(file)
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
    for name in list(loss):
            if name in names:
                assert name in list(models)

                # fig.suptitle('Aligning x-axis using sharex')
                ax1.plot( [loss[name][i] / loss[name][i - 1] for i in np.arange(1,len(n),1)], label=name)
                ax2.plot( range(len(k1_test)), loss[name], label=name)
                ax1.set_xticks([], [])
                ax2.set_xticks(range(len(k1_test)), range(len(k1_test)))
                ax1.set( ylabel='Rate')
                ax1.set_title('Error rates')
                ax2.set_title('Error values')



    plt.legend(loc="upper left")
    plt.xlabel(r'$k_{test}/k_0$')
    plt.ylabel(r'${ \mathrm{Error} }$')  #

    if save[0]:
     print(' saved as:errors_figure'+str(save[1])+'.eps')
     plt.savefig(Constants.FIGURES_PATH+'errors_figure'+save[1]+'.eps', format='eps',
                    bbox_inches='tight')

     plt.show()


# loop starts here
def solve_equation(names, n, x, t, time_steps, k1_test, k2_test):
    loss = dict((key, []) for key in list(names))

    for i in range(len(n)):
        test_constants = Constants(n[i], x[i], t[i], time_steps[i], k1_test[i], k2_test[i])
        create_test_data(test_constants)
        with open(path + 'test/test_data.pkl', 'rb') as file:
            test_data = pickle.load(file)

        data = {name: test_data[name][0] for name in list(test_data)}
        assert len(test_data['e'])==1

        data = {name: test_data[name][0] for name in list(test_data)}
        for name in list(loss):
            print('solve for model:  '+str(name))
            loss[name].append(loss_yee(name, models[name][0], models[name][1], models[name][2], data, test_constants))

    pickle.dump(loss, open(path + "figures/losses.pkl", "wb"))

#####################################
######################################
n = [41]*2
x=[1]*2
t = [2 / (36 * 2 ** 0.5)]*2
time_steps = [21]*2
k1_test = [[36]]*2
k2_test = [[36]]*2
names=[ 'Yee(4,0)', 'dl(2,1)', 'dl(2,3)']
# dr_calculator(names, save=('True','fig0000'))
# print(q)
error_print(names, n, x, t, time_steps, k1_test, k2_test, solve=True, save=('True','fig0000'))
# solve_equation(names, n, x, t, time_steps, k1_test, k2_test)
print(q)
######################################################
######################################################


# n = [ 21, 21,21,21]
# x = [1, 1,1,1]
# time_steps = [41, 81, 161,321]
# t = [ 2 / (9 * 2 ** 0.5), 4 / (9 * 2 ** 0.5), 8/ (9 * 2 ** 0.5), 16/ (9 * 2 ** 0.5)]
#
# k_test = [16 , 16,16,16]
# solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# k_test = [18 , 18,18,18]
# solve_equation(n, x, t, time_steps, k_test)
# # error_calculator(quotient=0, k_test=k_test)
# error_calculator(quotient=1, k_test=k_test)
#
# print(q)
###################
# figure eror quotient :
# dr_calculator()
# print(q)
n = [21, 41, 81, 161, 321]
t = [2 / (18 * 2 ** 0.5), 2 / (36 * 2 ** 0.5), 2 / (72 * 2 ** 0.5), 2 / (144 * 2 ** 0.5), 2 / (288 * 2 ** 0.5)]
x = [1, 1, 1, 1, 1]
time_steps = [21, 21, 21, 21, 21]
k_test = [18, 36, 72, 144, 288]
names=['dl(2,3)', 'Yee(4,0)', 'dl(4,1)']
error_print(names, n, x, t, time_steps, k1_test, k2_test, solve=True, save=('False','fig0000'))
solve_equation(n, x, t, time_steps, k_test)
solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# error_calculator(quotient=1, k_test=k_test)
print(q)
#################################


####################################
# figure eror quotient2 :
# var=-0.09053062
# n = [21, 41, 81, 161, 321]
# t = [2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5) ]
# x = [1, 1, 1, 1, 1]
# time_steps = [21, 41, 81, 161, 321]
# k_test = [18, 36, 72, 144, 288]
#
# solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# error_calculator(quotient=1, k_test=k_test)

#######################################
# figure eror figure 3 :
# var=-0.09053188
# n = [21, 81,81,81]*40
# t = [2 / (5 * 2 ** 0.5)]*40
# x = [1]*40
# time_steps = [21, 145,145,145]
# k_test = np.arange(20,40,1)
# # dr_calculator(n,x,t,time_steps,k_test)
# #
# solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# print(q)
#######################################
# print(q)
#
# k=23*2**0.5
# y=np.linspace(0,2/184/2**0.5,210)
# x=np.linspace(0,2/184/2**0.5,81)
# plt.scatter(x,np.cos((math.pi*x*k)))
# plt.plot(y,np.cos((math.pi*y*k)))
# plt.show()
# n = [161]*80
# t = [1 / (20 * 2 ** 0.5)] * 80
# time_steps = n
# k_test = np.arange(80,160,1)
# dr_calculator(n,t,time_steps,k_test)


#
# from drp import calculate_DRP
# from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2
# from DRP_multiple_networks.auxilary.aux_functions import relative_norm

# #
# import matplotlib.pyplot as plt
# # from DRP_multiple_networks import constants
#
#
# # x=calculate_DRP()
# # y=calculate_DRP2()
# # print(x)
# # print(y)
# # # print(Constants.CFL)
# # # # x=tf.constant([[1,2],[1,2]])
# # # # print(relative_norm(x,x ))
# # # # print(tf.math.pow(x,2))
# # X=np.arange(10, 19, 1)**2
# # Y=np.arange(1,20,1)**2
# # Z=Y
# # for x in X:
# #     for y in X:
# #         print(np.sqrt(x+y))
# #
# #
# # print(q)

# # # x=np.linspace(0,1,82)
# # # plt.plot(x,np.cos(math.pi*x*20))
# plt.show()
# # #
# # # print(Constants.CFL)
