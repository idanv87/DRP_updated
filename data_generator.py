import math
import pickle

import numpy as np

from constants import Constants
from utils import f_a

path = Constants.PATH


def generate_data(k1_train, k2_train):
    ex = []
    ey1 = []
    ey2=[]

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
           np.vstack(hx_x), np.vstack(hx_y1), np.vstack(hx_y2), np.vstack(hy_x), np.vstack(hy_y1), np.vstack(hy_y2), np.vstack(
        energy)


def create_test_data():
    k1_test = Constants.K1_TEST
    k2_test = Constants.K2_TEST
    ex, ey1, ey2, hx_x, hx_y1, hx_y2, hy_x, hy_y1, hy_y2, energy = generate_data(k1_test, k2_test)
    pickle.dump(ex.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex_test.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x_test.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x_test.pkl", "wb"))
    return 1


def create_validation_data():
    k1 = Constants.K1_VAL
    k2 = Constants.K2_VAL

    ex, ey1, ey2, hx_x, hx_y1, hx_y2, hy_x, hy_y1, hy_y2, energy = generate_data(k1, k2)
    pickle.dump(ex.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "ex_val.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "hx_x_val.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "hy_x_val.pkl", "wb"))
    pickle.dump(ey1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N * 2, Constants.N, 1)),
                open(path + "ey_val.pkl", "wb"))
    pickle.dump(hx_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2) * 2, Constants.N - 1, 1)),
                open(path + "hx_y_val.pkl", "wb"))
    pickle.dump(hy_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1) * 2, Constants.N - 2, 1)),
                open(path + "hy_y_val.pkl", "wb"))
    pickle.dump(energy.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, 1)), open(path + "energy_y_val.pkl", "wb"))

    return 1


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
    pickle.dump(ey1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N , Constants.N, 1)),
                open(path + "ey1.pkl", "wb"))
    pickle.dump(ey2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, Constants.N , Constants.N, 1)),
                open(path + "ey2.pkl", "wb"))
    pickle.dump(hx_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2) , Constants.N - 1, 1)),
                open(path + "hx_y1.pkl", "wb"))
    pickle.dump(hx_y2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 2) , Constants.N - 1, 1)),
                open(path + "hx_y2.pkl", "wb"))
    pickle.dump(hy_y1.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1) , Constants.N - 2, 1)),
                open(path + "hy_y1.pkl", "wb"))
    pickle.dump(hy_y2.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, (Constants.N - 1) , Constants.N - 2, 1)),
                open(path + "hy_y2.pkl", "wb"))
    pickle.dump(energy.reshape((len(k1) * len(k2) * Constants.TIME_STEPS, 1)), open(path + "energy_y.pkl", "wb"))
