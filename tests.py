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
# # k=17
# # x=np.linspace(0,1,21)
# # y=np.linspace(0,1,210)
# # scatter(x,sin(math.pi*k*x))
# # plot(y,sin(math.pi*k*y))
# # plt.show()
# # print(q)
# means=[]
# vars=[]
# ind=[]
# t, x, y = np.meshgrid(np.linspace(0, model_constants.T, model_constants.TIME_STEPS), model_constants.X1, model_constants.X2, indexing='ij')
# q=[15,17]
# E=[]
# Hx=[]
# Hy=[]
# for k1 in [13,15,17,11]:
#     # for k2 in np.arange(10,20,1):
#         k2=k1
#         omega = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
#         E.append(fE(t, x, y, k1, k2, omega)-tf.reduce_mean(fE(t*0, x, y, k1, k2, omega)))
#         Hx.append(fHX(t + model_constants.DT / 2, x, y, k1, k2, omega, model_constants.DX)-tf.reduce_mean(fHX(t + model_constants.DT / 2, x, y, k1, k2, omega, model_constants.DX)))
#         Hy.append(fHY(t + model_constants.DT / 2, x, y, k1, k2, omega, model_constants.DX)-tf.reduce_mean(fHY(t + model_constants.DT / 2, x, y, k1, k2, omega, model_constants.DX)))
#
#
# s1=np.mean(np.array([tf.math.multiply(E[k],E[k]) for k in range(len(E))]))
# s2=np.mean(np.array([Hx[k]*Hx[k] for k in range(len(Hx))]))
# s3=np.mean(np.array([Hy[k]*Hy[k] for k in range(len(Hy))]))
# print(s1+s2+s3)
#
#   # +np.mean([tf.reduce_prod(Hx[k],Hx[k]) for k in range(len(Hx))])+np.mean([tf.reduce_prod(Hy[k],Hy[k]) for k in range(len(Hy))])
# print(qq)
#plt.plot(np.arange(10,20,1),vars)
# plt.scatter(q,[q[k]*0 for k in range(len(q))])
# plt.plot(np.arange(10,21,1),[vars[k] for k in np.arange(len(vars))])
# plt.scatter(np.arange(10,21,1),np.arange(10,21,1)*0)
# plt.plot([abs(vars[k]-sum(vars)/len(vars)) for k in np.arange(len(vars))])


# for k_test in [10,11,12,13,14,15,16,17,18,19]:
#    var_constants=Constants(model_constants.N, 1, model_constants.T,model_constants.TIME_STEPS, k_test)
#    create_test_data(var_constants)
#    with open(path + 'test/test_data.pkl', 'rb') as file:
#       test_data = pickle.load(file)
#    # l+=tf.math.reduce_mean(test_data['e'][0])+tf.math.reduce_mean(test_data['hx'][0])+tf.math.reduce_mean(test_data['hy'][0])
#    # m+=tf.math.reduce_variance(test_data['e'][0])+tf.math.reduce_variance(test_data['hx'][0])+tf.math.reduce_variance(test_data['hy'][0])
#    means.append(tf.math.reduce_mean(test_data['e'][0])+tf.math.reduce_mean(test_data['hx'][0])+tf.math.reduce_mean(test_data['hy'][0]))
#    vars.append(tf.math.reduce_variance(test_data['e'][0])+tf.math.reduce_variance(test_data['hx'][0])+tf.math.reduce_variance(test_data['hy'][0]))
#
# print(len(means))
# print(q)
# print(l/len(means))



   # print(tf.math.reduce_variance(test_data['e'][0])+tf.math.redcue_variance(test_data['hx'][0])+tf.math.reduce_variance(test_data['hy'][0]))
var=calculate_DRP2()
# print(var)


models = {'yee': [0.], 'yee4': [-1 / 24], 'drp(2,1)': [var], 'dl(2,1)': [], 'dl(2,3)_all': [], 'dl(2,3)': [],
          'dl(4,1)': [], 'model_test':[]}

for name in ['dl(2,1)', 'model_test', 'dl(2,3)', 'dl(4,1)', 'dl(2,3)_all']:
    saving_path = path + 'Experiment_' + name + '_details/'
    model = keras.models.load_model(saving_path + 'model.pkl',
                                    custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
    model.load_weights(
        saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()  # print(model1.trainable_weights)
    models[name] = model.trainable_weights


l_yee = []
l_yee4 = []
l_drp21 = []
l_dl21 = []
l_dl23_all = []
l_dl23 = []
l_dl41 = []

dr_yee = []
dr_yee4 = []
dr_drp21 = []
dr_dl21 = []
dr_dl23_all = []
dr_dl23 = []


def dr_calculator_extended():
    X = np.linspace(math.pi / 2, math.pi, model_constants.N * 50)
    x, y = np.meshgrid(X, X, indexing='ij')
    loss_drp_extended(model_constants,X,X, models['dl(2,3)'][0], models['dl(2,3)'][1], models['dl(2,3)'][2])

#

def dr_calculator():
    # a=[9/8, 1-3*var, 1 - 3 * models['dl(2,1)'][0] ]
    par1,par2, par3=models['dl(2,3)_all']
    X = np.linspace(0.5*math.pi , math.pi, model_constants.N*50)
    # X=model_constants.DX*np.arange(10,20,1)*math.pi
    x, y = np.meshgrid(X, X, indexing='ij')
    # plt.plot(X/math.pi,loss_drp(model_constants, X, X,a[0]), label='yee4')
    plt.plot(X/math.pi, loss_drp_extended(model_constants, X, X, 0, var,0), label='drp21')
    # plt.plot(X / math.pi, loss_drp(model_constants, X, X, 1-3*var), label='drp21')

    plt.plot(X/math.pi, loss_drp_extended(model_constants, X, X, 0, models['dl(2,1)'][0],0), label='dl21')
    plt.plot(X / math.pi, loss_drp_extended(model_constants, X, X, par1,par2,par3), label='dl23_all')


    plt.plot(X / math.pi, loss_drp_extended(model_constants, X, X, models['dl(2,3)'][0],models['dl(2,3)'][1],models['dl(2,3)'][2] ), label='dl23')

    plt.legend(loc="upper left")
    plt.xlabel(r'$\frac{\xi}{\pi}$')
    plt.ylabel('dispersion error')


    #
    # plt.savefig('/Users/idanversano/documents/papers/drp/figures/dr_loss.eps', format='eps',
    #             bbox_inches='tight')
    plt.show()
    return 1



def error_calculator(quotient, k_test):
    loss = {'yee': None, 'yee4': None, 'drp21': None, 'dl21': None, 'dl23_all': None, 'dl23': None, 'dl41': None}

    for name in list(loss):
        with open(path + 'figures/l_' + name + '.pkl', 'rb') as file:
            loss[name] = pickle.load(file)
            if name in [  'drp21', 'dl21',
                 'dl23', 'dl23_all'
                , 'dl23_all', 'dl41'
                        ]:
                if quotient:
                    plt.plot([2,3,4],[loss[name][i] / loss[name][i - 1] for i in [1, 2, 3]], label=name)
                else:
                    plt.plot(np.arange(len(k_test))+1, loss[name], label=name)




    plt.legend(loc="upper left")
    plt.xlabel(r'$k_{test}/k_0$')
    # plt.xlabel(r'$T/T_0$')
    plt.ylabel(r'${ \mathrm{Error} }$')  #
    if quotient:
        pass

        # plt.savefig('/Users/idanversano/documents/papers/drp/figures/error_quotient.eps', format='eps',
        #             bbox_inches='tight')
        #  plt.savefig('/Users/idanversano/documents/papers/drp/figures/time_quotient.eps', format='eps',
        #            bbox_inches='tight')
    else:
        pass

        # plt.savefig('/Users/idanversano/documents/papers/drp/figures/error_doubling.eps', format='eps',
        #             bbox_inches='tight')

        #  plt.savefig('/Users/idanversano/documents/papers/drp/figures/time_doubling.eps', format='eps',
        #            bbox_inches='tight')

    plt.show()


# loop starts here
def solve_equation(n, x, t, time_steps, k_test):
    for i in range(len(n)):
        test_constants = Constants(n[i], x[i], t[i], time_steps[i], k_test[i])
        create_test_data(test_constants)
        with open(path + 'test/test_data.pkl', 'rb') as file:
            test_data = pickle.load(file)

        data = {name: test_data[name][0] for name in list(test_data)}
        for i in range(len(test_data['e'])):
            data = {name: test_data[name][i] for name in list(test_data)}
            l_yee.append(loss_yee('yee', 0., 0., 0., data, test_constants))
            l_yee4.append(loss_yee('yee4', 0., -1 / 24, 0., data, test_constants))
            l_drp21.append(loss_yee('drp(2,1)', 0., var, 0., data, test_constants))
            l_dl21.append(loss_yee('dl(2,1)', 0., models['dl(2,1)'][0], 0., data, test_constants))
            l_dl23_all.append(loss_yee('dl(2,3)_all', models['dl(2,3)_all'][0], models['dl(2,3)_all'][1], models['dl(2,3)_all'][2], data, test_constants))

            l_dl23.append(loss_yee('dl(2,3)', models['dl(2,3)'][0], models['dl(2,3)'][1], models['dl(2,3)'][2], data,
                                   test_constants))
            l_dl41.append(
                loss_yee('dl(4,1)', models['dl(4,1)'][0], (16 * models['dl(4,1)'][0] - 1) / 24,
                         -models['dl(4,1)'][0] / 3,
                         data, test_constants))

    pickle.dump(l_yee, open(path + "figures/l_yee.pkl", "wb"))
    pickle.dump(l_yee4, open(path + "figures/l_yee4.pkl", "wb"))
    pickle.dump(l_drp21, open(path + "figures/l_drp21.pkl", "wb"))
    pickle.dump(l_dl21, open(path + "figures/l_dl21.pkl", "wb"))
    pickle.dump(l_dl23_all, open(path + "figures/l_dl23_all.pkl", "wb"))
    pickle.dump(l_dl23, open(path + "figures/l_dl23.pkl", "wb"))
    pickle.dump(l_dl41, open(path + "figures/l_dl41.pkl", "wb"))
###########################


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
t = [2 / (18 * 2 ** 0.5), 2 / (36 * 2 ** 0.5), 2 / (72 * 2 ** 0.5), 2 / (144 * 2 ** 0.5), 2 / (288 * 2 ** 0.5) ]
x = [1, 1, 1, 1, 1]
time_steps = [21, 21, 21, 21, 21]
k_test = [18, 36, 72, 144, 288]
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
