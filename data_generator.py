import math
import pickle

import numpy as np

from constants import Constants
from utils import f_a

path = Constants.PATH


def generate_data(k1_train, k2_train):
    ex = []
    ey1 = []
    ey2 = []

    hx_x = []
    hy_x = []
    hx_y1 = []

    hx_y2 = []
    hy_y1 = []
    hy_y2 = []

    energy = []
    for k1 in k1_train:
        for k2 in k2_train:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
            for n in range(2, Constants.TIME_STEPS + 2):
                f0 = f_a(c, n - 2, k1, k2)
                f1 = f_a(c, n - 1, k1, k2)
                f2 = f_a(c, n, k1, k2)

                ex.append(f0[0])
                ey1.append(f1[0])
                ey2.append(f2[0])

                hx_x.append(f0[1])
                hx_y1.append(f1[1])
                hx_y2.append(f2[1])

                hy_x.append(f0[2])
                hy_y1.append(f1[2])
                hy_y2.append(f2[2])

                energy.append(f1[3])

    return np.vstack(ex), np.vstack(ey1), np.vstack(ey2), \
           np.vstack(hx_x), np.vstack(hx_y1), np.vstack(hx_y2), np.vstack(hy_x), np.vstack(hy_y1), np.vstack(
        hy_y2), np.vstack(
        energy)


def create_test_data():
    k1_test = Constants.K1_TEST
    k2_test = Constants.K2_TEST
    #ex, hx_x, hy_x = generate_test_data(k1_test, k2_test)
    ex, ey1, ey2, hx_x, hx_y1, hx_y2, hy_x, hy_y1, hy_y2, energy  = generate_data(k1_test, k2_test)

    pickle.dump(ex.reshape((Constants.TEST_NUM  * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex_test.pkl", "wb"))
    pickle.dump(hx_x.reshape((Constants.TEST_NUM  * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x_test.pkl", "wb"))
    pickle.dump(hy_x.reshape((Constants.TEST_NUM  * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x_test.pkl", "wb"))
    return 1


def generate_test_data(k1_test, k2_test):
    e = []
    hx = []
    hy = []

    ex = np.zeros((Constants.TIME_STEPS * Constants.N, Constants.N))
    hx_x = np.zeros((Constants.TIME_STEPS * (Constants.N-2), Constants.N-1))
    hy_x = np.zeros((Constants.TIME_STEPS * (Constants.N-1), Constants.N-2))

    for k1 in k1_test:
        for k2 in k2_test:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
            for n in range(2, Constants.TIME_STEPS + 2):
                f0 = f_a(c, n - 2, k1, k2)
                e.append(f0[0])
                hx.append(f0[1])
                hy.append(f0[2])
    E = np.vstack(e)
    Hx = np.vstack(hx)
    Hy = np.vstack(hy)
    e=[]
    hx=[]
    hy=[]
    for i in range(Constants.TEST_NUM):
        a = np.random.rand(1, len(Constants.K1_TEST) * len(Constants.K2_TEST))
        print(a)
        for j in range(len(Constants.K1_TEST) * len(Constants.K2_TEST)):
            ex += a[0, j] * E[j*Constants.TIME_STEPS*Constants.N:Constants.TIME_STEPS * Constants.N*(j+1), :]
            hx_x += a[0, j] * Hx[j*Constants.TIME_STEPS*(Constants.N-2):Constants.TIME_STEPS * (Constants.N-2)*(j+1), :]
            hy_x += a[0, j] * Hy[j*Constants.TIME_STEPS*(Constants.N-1):Constants.TIME_STEPS * (Constants.N-1)*(j+1), :]
        e.append(ex)
        hx.append(hx_x)
        hy.append(hy_x)

    return np.vstack(e), np.vstack(hx), np.vstack(hy)


if __name__ == "__main__":
    print("generating data")
    k1 = Constants.K1_TRAIN
    k2 = Constants.K2_TRAIN

    ex, ey1, ey2, hx_x, hx_y1, hx_y2, hy_x, hy_y1, hy_y2, energy = generate_data(k1, k2)

    pickle.dump(ex.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x.pkl", "wb"))
    pickle.dump(ey1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ey1.pkl", "wb"))
    pickle.dump(ey2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ey2.pkl", "wb"))
    pickle.dump(hx_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2), Constants.N - 1, 1)),
                open(path + "hx_y1.pkl", "wb"))
    pickle.dump(hx_y2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2), Constants.N - 1, 1)),
                open(path + "hx_y2.pkl", "wb"))
    pickle.dump(hy_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1), Constants.N - 2, 1)),
                open(path + "hy_y1.pkl", "wb"))
    pickle.dump(hy_y2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1), Constants.N - 2, 1)),
                open(path + "hy_y2.pkl", "wb"))
    pickle.dump(energy.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, 1)), open(path + "energy_y.pkl", "wb"))
