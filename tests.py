import math
import pickle
import time
import matplotlib.pyplot as plt

from constants import Constants
import numpy as np
from data_generator import create_test_data

from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import numpy as np

# simulated data
dataset = datasets.make_classification(n_samples=10000, n_features=20, n_informative=5,
                                       n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2,
                                       weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                                       scale=1.0, shuffle=True, random_state=None)

X = dataset[0]
y = dataset[1]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []

for train, test in kfold.split(X, y):
    # Define a Deep Learning Model
    model = Sequential()
    model.add(Dense(38, input_dim=20,
                    kernel_regularizer=l2(0.001),  # weight regularizer
                    kernel_initializer=VarianceScaling(),  # initializer
                    activation='tanh'))
    model.add(Dense(25,
                    kernel_regularizer=l2(0.01),  # weight regularizer
                    kernel_initializer=VarianceScaling(),  # initializer
                    activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the Model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['acc'])

    # Train the Model
    model.fit(X[train], y[train], epochs=50, batch_size=25, verbose=0,
              validation_data=(X[test], y[test]))

    # evaluate the model
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print(q)


path=Constants.PATH

d=dict()
d.setdefault('a',1)
d.setdefault('b',2)

d.__setitem__('a',d['a']+2*d['b'])
print(d)





# print(l0)            # tracemalloc.start()
#             generate_basis(kx, ky,'train')
#         #  print(tracemalloc.get_traced_memory())
#         # tracemalloc.stop()