import math
import pickle


import numpy as np

from constants import Constants
from utils import f_a

path=Constants.PATH

def generate_data(k1_train, k2_train):
    ex = []
    ey = []
    hx_x = []
    hy_x = []
    hx_y = []
    hy_y = []
    inte=[]
    inth=[]
    for k1 in k1_train:
        for k2 in k2_train:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))

            for n in range(2, Constants.TIME_STEPS + 2):
                f0 = f_a(c, n - 2, k1, k2)
                f1 = f_a(c, n - 1, k1, k2)
                f2 = f_a(c, n, k1, k2)
                ex.append(f0[0])
                ey.append(np.vstack((f1[0], f2[0])))
                hx_x.append(f0[1])
                hx_y.append(np.vstack((f1[1], f2[1])))
                hy_x.append(f0[2])
                hy_y.append(np.vstack((f1[2], f2[2])))
                inte.append(f1[3])
                inth.append(f1[4])



    return np.vstack(ex), np.vstack(ey), np.vstack(hx_x), np.vstack(hx_y), np.vstack(hy_x), np.vstack(hy_y), np.vstack(inte), np.vstack(inth)

if __name__=='main':
    k1 = Constants.K1_TRAIN
    k2 = Constants.K2_TRAIN
    ex, ey, hx_x, hx_y, hy_x, hy_y, inte, inth = generate_data(k1, k2)

    pickle.dump(ex.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x.pkl", "wb"))
    pickle.dump(ey.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N * 2, Constants.N, 1)),
                open(path + "ey.pkl", "wb"))
    pickle.dump(hx_y.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2) * 2, Constants.N - 1, 1)),
                open(path + "hx_y.pkl", "wb"))
    pickle.dump(hy_y.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1) * 2, Constants.N - 2, 1)),
                open(path + "hy_y.pkl", "wb"))
    pickle.dump(inte.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, 1)), open(path + "inte.pkl", "wb"))
    pickle.dump(inth.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, 1)), open(path + "inth.pkl", "wb"))



def create_test_data():
    k1_test = Constants.K1_TEST
    k2_test = Constants.K2_TEST
    ex, ey, hx_x, hx_y, hy_x, hy_y, inte, inth = generate_data(k1_test, k2_test)
    pickle.dump(ex.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex_test.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x_test.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x_test.pkl", "wb"))
    return 1

