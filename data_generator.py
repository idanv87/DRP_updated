import math
import pickle
import time
import os

import numpy as np

from constants import Constants
from utils import fE, fHX, fHY, f_a

path = Constants.PATH


def generate_train_data(k1, k2):
    c = Constants.PI * np.sqrt(k1 ** 2 + k2 ** 2)

    if k1 == k2:
        Energy = [1] * Constants.TIME_STEPS
    else:
        Energy = [1 / 2] * Constants.TIME_STEPS

    t, x, y = np.meshgrid(np.linspace(0, Constants.T, Constants.TIME_STEPS), Constants.X1, Constants.X2, indexing='ij')

    c = Constants.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

    FE = np.sin(Constants.PI * k1 * x) * np.sin(Constants.PI * k2 * y) + \
         np.sin(Constants.PI * k2 * x) * np.sin(Constants.PI * k1 * y)

    FHX = -Constants.PI * k2 * np.sin(Constants.PI * k1 * x) * np.cos(
        Constants.PI * k2 * (y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * x) * np.cos(
        Constants.PI * k1 * (y + Constants.DX / 2))

    FHY = Constants.PI * k1 * np.cos(Constants.PI * k1 * (x + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (x + Constants.DX / 2)) * np.sin(
        Constants.PI * k1 * y)

    isExist = os.path.exists(path + 'base_functions/')
    if not isExist:
        os.makedirs(path + 'base_functions/')

    pickle.dump(
        [fE(FE, 0, t, c), fE(FE, 1, t, c), fE(FE, 2, t, c), fHX(FHX, 1, t, c), fHX(FHX, 3, t, c), fHX(FHX, 5, t, c),
         fHY(FHY, 1, t, c), fHY(FHY, 3, t, c), fHY(FHY,
                                                   5, t, c), np.vstack(Energy)]
        , open(path + 'base_functions/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_train.pkl', "wb"))

    return 1


def create_train_data(options='lt'):
    sol = {'ex': [], 'ey1': [], 'ey2': [], 'hx_x': [], 'hx_y1': [], 'hx_y2': [], 'hy_x': [], 'hy_y1': [],
           'hy_y2': [],
           'energy': []}

    if options == 'lt':
        for i in np.arange(Constants.TRAIN_NUM):

            dict = {'ex': [], 'ey1': [], 'ey2': [], 'hx_x': [], 'hx_y1': [], 'hx_y2': [], 'hy_x': [], 'hy_y1': [],
                    'hy_y2': [],
                    'energy': []}
            for k1 in Constants.K1_TRAIN:
                for k2 in Constants.K2_TRAIN:
                    with open(path + 'base_functions/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_train.pkl',
                              'rb') as file:
                        l = pickle.load(file)

                    a = np.random.rand(1)

                    [dict[list(dict)[k]].append(a * l[k]) for k in np.arange(len(dict))]

            [sol[list(sol)[k]].append(sum(dict[list(dict)[k]])) for k in np.arange(len(sol))]
        else:
            for k1 in Constants.K1_TRAIN:
                for k2 in Constants.K2_TRAIN:
                    with open(path + 'base_functions/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_train.pkl',
                              'rb') as file:
                        l = pickle.load(file)
                    [sol[list(sol)[k]].append(l[k]) for k in np.arange(len(sol))]

    isExist = os.path.exists(path + 'train/')
    if not isExist:
         os.makedirs(path + 'train/')
    pickle.dump(sol, open(path+"train/train_data.pkl", "wb"))

    return sol


def generate_test_data(k1_test, k2_test):
    ex = []
    hx_x = []
    hy_x = []
    for k1 in k1_test:
        for k2 in k2_test:
            c = Constants.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

            for n in range(2, Constants.TIME_STEPS + 2):
                f0 = f_a(c, n - 2, k1, k2)
                ex.append(f0[0])
                hx_x.append(f0[1])
                hy_x.append(f0[2])

    return np.vstack(ex), np.vstack(hx_x), np.vstack(hy_x)


def create_test_data():
    k1_test = Constants.K1_TEST
    k2_test = Constants.K2_TEST
    ex, hx_x, hy_x = generate_test_data(k1_test, k2_test)

    isExist = os.path.exists(path + 'test/')
    if not isExist:
        os.makedirs(path + 'test/')

    pickle.dump(ex.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "test/ex_test.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "test/hx_x_test.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "test/hy_x_test.pkl", "wb"))
    return 1




if __name__ == "__main__":
    for kx in Constants.K1_TRAIN:
        for ky in Constants.K2_TRAIN:
            generate_train_data(kx, ky)
    create_train_data()