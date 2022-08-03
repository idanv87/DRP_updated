import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.axis
import numpy as np

from DRP_multiple_networks.constants import Constants

path=Constants.PATH
d=dict()
for name in ['yee','fourth', 'model', 'drp']:
   with open(path + 'figures/loss_' +name+'.pkl', 'rb') as file:
       d.setdefault(name, pickle.load(file))


k=np.sqrt(Constants.K1_TEST**2+Constants.K2_TEST**2)




plt.plot(k, d['yee'], "b", label="Yee")
plt.plot(k, d['fourth'], "r", label="4th")
plt.plot(k, d['model'], "g", label="DL2", linestyle='dashed')
plt.plot(k ,d['drp'], "-d", label="DRP")
#plt.plot([(l_drp[k]/l_model[k]) for k in range(len(l_drp))], "b", label="gpa")
plt.legend(loc="upper left")
plt.xlabel(r'${\bf \|\vec{k}\|}$')
plt.ylabel(r'${\bf \mathrm{Error}}$' )
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.show()