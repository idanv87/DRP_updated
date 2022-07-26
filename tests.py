import pickle
import time

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants
from utils import *
from data_generator import create_test_data



n=2
a=np.random.rand(n,n)
b=np.random.rand(n,n)
c=np.random.rand(n,n)


# l=[a,b,c]
z=[1,2,3]
l=np.hstack(np.array([a,b,c]))

z=np.tile(np.repeat(np.array(z), 2),(2,1))
print(z.shape)



start_time = time.time()
#for k in np.arange(3):
#  l0.append(z[k]*l[k])
#l2=map(lambda k: z[k]*l[k], [0,1,2])
# l=[a*b for a,b in zip(l,z)]
z*l
print("--- %s seconds ---" % (time.time() - start_time))
# print(l)
# print(l0)