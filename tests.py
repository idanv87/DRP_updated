import pickle

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants
from utils import *
from data_generator import create_test_data


a=np.random.rand(2,2)
b=np.random.rand(2,2)
c=np.random.rand(2,2)
l=[a,b,c]
z=np.vstack(l).reshape(3,2,2,1)
print(z[1,:,:,0])
print(" ")
print(a)
print(b)

