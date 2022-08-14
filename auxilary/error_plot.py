import pickle
import math

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


from DRP_multiple_networks.utils import loss_yee3
from DRP_multiple_networks.constants import Constants
from DRP_multiple_networks.utils import custom_loss, custom_loss3
from DRP_multiple_networks.data_generator import create_test_data
from DRP_multiple_networks.drp import calculate_DRP

path=Constants.PATH
create_test_data()
with open(path + 'test/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)


for i in range(1):
    data = {name: test_data[name][i] for name in list(test_data)}
    divergence, energy, lt_E, lt_H=loss_yee3('4order', 0., -1/24, data)
t=np.linspace(Constants.DT, Constants.DT*(Constants.TIME_STEPS-3),Constants.TIME_STEPS-3)
# plt.plot(np.array(energy)-np.sin(math.pi*np.sqrt(2)*(t+Constants.DT/2))**2, label='energy')

# plt.plot(np.array(lt_H)*np.array(lt_E))
# plt.plot(np.array(lt_E),label='E')
# plt.plot(np.array(lt_H), label='H')
plt.plot(np.log10(abs(np.array(energy)-0.5)), label='energy')
plt.legend(loc="upper left")
plt.xlabel(r'$t$')
plt.ylabel(r'${ \mathrm{Error} }$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title('Error over time')
plt.savefig('/Users/idanversano/documents/papers/drp/figures/error_time2.eps', format='eps',
            bbox_inches='tight')
plt.show()
