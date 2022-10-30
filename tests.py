import numpy as np
import math

from drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2
from DRP_multiple_networks.auxilary.aux_functions import relative_norm
import tensorflow as tf
from DRP_multiple_networks.constants import Constants
import matplotlib.pyplot as plt

x=calculate_DRP()
y=calculate_DRP2()
print(x)
print(y)
# print(Constants.CFL)
# # x=tf.constant([[1,2],[1,2]])
# # print(relative_norm(x,x ))
# # print(tf.math.pow(x,2))
# x=np.linspace(0,1,21)
# plt.scatter(x,np.cos(math.pi*x*20))
# x=np.linspace(0,1,82)
# plt.plot(x,np.cos(math.pi*x*20))
# plt.show()
#
# print(Constants.CFL)
