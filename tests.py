import math
import pickle
import time
import matplotlib.pyplot as plt

from constants import Constants
import numpy as np
from data_generator import create_test_data
path=Constants.PATH

x=np.linspace(0,0.5,63)
y=np.cos(math.pi*30*x)
plt.plot(x,y)
plt.show()
print(q)


x=np.random.rand(10,1000,1000)
y=np.random.rand(10,1000,1000)
l=[x,y]
time_start=time.time()
# pickle.dump(x, open(path+'file1.pkl', "wb"))
# pickle.dump(y, open(path+'file2.pkl', "wb"))
pickle.dump(l, open(path+'file3.pkl', "wb"))


print(time.time()-time_start)
# print(f.shape)
# print(f[0].shape)
#print(np.squeeze(f)-g)






# print(l0)