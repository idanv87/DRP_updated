import numpy as np
import math

from drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2
from DRP_multiple_networks.auxilary.aux_functions import relative_norm
import tensorflow as tf
# from DRP_multiple_networks.constants import Constants
import matplotlib.pyplot as plt
# from DRP_multiple_networks import constants


# x=calculate_DRP()
# y=calculate_DRP2()
# print(x)
# print(y)
# # print(Constants.CFL)
# # # x=tf.constant([[1,2],[1,2]])
# # # print(relative_norm(x,x ))
# # # print(tf.math.pow(x,2))
# X=np.arange(10, 19, 1)**2
# Y=np.arange(1,20,1)**2
# Z=Y
# for x in X:
#     for y in X:
#         print(np.sqrt(x+y))
#
#
# print(q)
k=7
y=np.linspace(0,1,210)
x=np.linspace(0,1,37)
plt.scatter(x,np.cos((math.pi*x*k)))
plt.plot(y,np.cos((math.pi*y*k)))
# # x=np.linspace(0,1,82)
# # plt.plot(x,np.cos(math.pi*x*20))
plt.show()
# #
# # print(Constants.CFL)
