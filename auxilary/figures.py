import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.axis
import numpy as np

from DRP_multiple_networks.constants import Constants

path=Constants.PATH
d=dict()

if False:
    for name in ['model1', 'model2', 'drp']:
        with open(path + 'figures/loss_time_' + name + '.pkl', 'rb') as file:
            d.setdefault(name, pickle.load(file))
    k = [str(x) for x in Constants.K1_TEST]
    plt.plot( d['model1'], "g", label="DL2", linestyle='dashed')
    #plt.plot(k, d['model2'], "b", label="DL2T", linestyle='dashed')
    plt.plot( d['drp'], "-d", label="DRP")
    # plt.plot([(l_drp[k]/l_model[k]) for k in range(len(l_drp))], "b", label="gpa")
    plt.legend(loc="upper left")
    plt.xlabel(r'${ (k_x,20)}$')
    plt.ylabel(r'${ \mathrm{Error} }$')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(r'${\bf L^{\infty}- \mathrm{Error} \ \mathrm{over}  \ |\vec{k}|}$')
    plt.savefig('/Users/idanversano/documents/papers/drp/figures/fig_T.eps', format='eps', bbox_inches='tight')
    plt.show()



if False:
    for name in ['yee', 'fourth', 'model1', 'model2', 'drp']:
        with open(path + 'figures/loss_' + name + '.pkl', 'rb') as file:
            d.setdefault(name, pickle.load(file))
    k = [str(x) for x in Constants.K1_TEST]
    # plt.plot(k, d['yee'], "b", label="Yee")
    plt.plot(k, d['fourth'], "r", label="Yee4")
    plt.plot(k, d['model1'], "g", label="DL2", linestyle='dashed')
    #plt.plot(k, d['model2'], "b", label="DL2T", linestyle='dashed')
    plt.plot(k, d['drp'], "-d", label="DRP")
    # plt.plot([(l_drp[k]/l_model[k]) for k in range(len(l_drp))], "b", label="gpa")
    plt.legend(loc="upper left")
    plt.xlabel(r'${ (k_x,20)}$')
    plt.ylabel(r'${ \mathrm{Error} }$')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(r'${\bf L^{\infty}- \mathrm{Error} \ \mathrm{over}  \ |\vec{k}|}$')
    plt.savefig('/Users/idanversano/documents/papers/drp/figures/fig_k.eps', format='eps', bbox_inches='tight')
    plt.show()
if True:
    for name in ['yee', 'fourth', 'model1', 'model2', 'drp']:
        with open(path + 'figures/loss_time_' + name + '.pkl', 'rb') as file:
            d.setdefault(name, pickle.load(file))

    #plt.plot( d['yee'], "b", label="Yee")
    plt.plot( d['fourth'], "r", label="Yee4")
    plt.plot( d['model1'], "g", label="DL2", linestyle='dashed')
    #plt.plot(d['model2'], "b", label="DL2T", linestyle='dashed')
    plt.plot( d['drp'], "-d", label="DRP")
    # plt.plot([(l_drp[k]/l_model[k]) for k in range(len(l_drp))], "b", label="gpa")
    plt.legend(loc="upper left")
    plt.xlabel(r'$t$')
    plt.ylabel(r'${ \mathrm{Error} }$')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(r'${\bf L^{\infty}- \mathrm{Error} \ \mathrm{over}  \ \mathrm{time}}$')
    plt.savefig('/Users/idanversano/documents/papers/drp/figures/low_res_more_more_time_high.eps', format='eps', bbox_inches='tight')
    plt.show()
