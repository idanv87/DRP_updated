import pickle
import time
import os
import tracemalloc

import numpy as np

from constants import Constants
from utils import fE, fHX, fHY

C = Constants()
path = C.PATH


class Names(dict):
    def __init__(self, xy, train_or_test):
        assert xy in ['X', 'Y']
        assert train_or_test in ['train', 'test']

        super().__init__()
        saving_folder = path + 'base_functions/' + train_or_test + '/'
        isExist = os.path.exists(saving_folder)
        if not isExist:
            os.makedirs(saving_folder)
        self.path = saving_folder + xy + '_'

        if xy == 'X':
            [self.__setitem__(name, []) for name in ['e_x', 'hx_x', 'hy_x']]
        else:
            for i in range(C.NUM_OUTPUT):
                [self.__setitem__(name + '_y' + str(i + 1), []) for name in ['e', 'hx', 'hy']]
            self.__setitem__('energy', [])


def generate_basis(k1, k2, name):
    assert name in ['train', 'test']
    X_train = Names('X', name)
    Y_train = Names('Y', name)

    if k1 == k2:
        Y_train.__setitem__('energy', np.vstack([1.] * C.TIME_STEPS))
    else:
        Y_train.__setitem__('energy', np.vstack([1 / 2] * C.TIME_STEPS))

    t, x, y = np.meshgrid(np.linspace(0, C.T, C.TIME_STEPS), C.X1, C.X2, indexing='ij')

    c = C.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

    FE = np.sin(C.PI * k1 * x) * np.sin(C.PI * k2 * y) + \
         np.sin(C.PI * k2 * x) * np.sin(C.PI * k1 * y)
    X_train.__setitem__('e_x', fE(FE, 0, t, c))
    [Y_train.__setitem__('e_y' + str(i + 1), fE(FE, i + 1, t, c)) for i in range(C.NUM_OUTPUT)]
    del FE

    FHX = -C.PI * k2 * np.sin(C.PI * k1 * x) * np.cos(
        C.PI * k2 * (y + C.DX / 2)) - C.PI * k1 * np.sin(
        C.PI * k2 * x) * np.cos(
        C.PI * k1 * (y + C.DX / 2))
    X_train.__setitem__('hx_x', fHX(FHX, 1, t, c))
    [Y_train.__setitem__('hx_y' + str(i + 1), fHX(FHX, 2 * (i + 1) + 1, t, c)) for i in range(C.NUM_OUTPUT)]
    del FHX

    FHY = C.PI * k1 * np.cos(C.PI * k1 * (x + C.DX / 2)) * np.sin(
        C.PI * k2 * y) + C.PI * k2 * np.cos(
        C.PI * k2 * (x + C.DX / 2)) * np.sin(
        C.PI * k1 * y)

    X_train.__setitem__('hy_x', fHY(FHY, 1, t, c))
    [Y_train.__setitem__('hy_y' + str(i + 1), fHY(FHY, 2 * (i + 1) + 1, t, c)) for i in range(C.NUM_OUTPUT)]

    del FHY

    pickle.dump(X_train, open(X_train.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', "wb"))
    del X_train
    pickle.dump(Y_train, open(Y_train.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', "wb"))

    return 1


def create_train_data(options='lt'):
    train_x = Names('X','train')
    train_y = Names('Y','train')

    if options == 'lt':

        for i in np.arange(C.TRAIN_NUM):
            dictx = {name: 0. for name in list(train_x)}
            dicty = {name: 0. for name in list(train_y)}
            for k1 in C.K1_TRAIN:
                for k2 in C.K2_TRAIN:
                    a = np.random.rand(1) / (len(C.K1_TRAIN) * len(C.K2_TRAIN))
                    with open(train_x.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', 'rb') as file:
                        lx = pickle.load(file)
                    [dictx.__setitem__(name, dictx[name] + a * lx[name]) for name in list(train_x)]
                    del lx

                    with open(train_y.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', 'rb') as file:
                        ly = pickle.load(file)
                    [dicty.__setitem__(name, dicty[name] + a * ly[name]) for name in list(train_y)]
                    del ly

            [train_x[name].append(dictx[name]) for name in list(train_x)]
            [train_y[name].append(dicty[name]) for name in list(train_y)]


    else:
        for k1 in C.K1_TRAIN:
            for k2 in C.K2_TRAIN:
                with open(train_x.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', 'rb') as file:
                    lx = pickle.load(file)
                [train_x[name].append(lx[name]) for name in list(train_x)]
                del lx
                with open(train_y.path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', 'rb') as file:
                    ly = pickle.load(file)
                [train_y[name].append(ly[name]) for name in list(train_y)]
                del ly

    isExist = os.path.exists(path + 'train/')
    if not isExist:
        os.makedirs(path + 'train/')


    pickle.dump(C, open(path + "train/train_constants.pkl", "wb"))
    pickle.dump(dict(train_x), open(path + "train/X_train.pkl", "wb"))
    pickle.dump(dict(train_y), open(path + "train/Y_train.pkl", "wb"))

    return 1


def create_test_data(k1_test=C.K1_TEST, k2_test=C.K2_TEST):
    for kx in k1_test:
        for ky in k2_test:
            generate_basis(kx, ky, 'test')

    sol = {'e_x': [], 'hx_x': [], 'hy_x': []}
    for k1 in k1_test:
        for k2 in k2_test:
            with open(path + 'base_functions/test/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', 'rb') as file:
                l = pickle.load(file)
            [sol[name].append(l[name]) for name in list(sol)]

    isExist = os.path.exists(path + 'test/')
    if not isExist:
        os.makedirs(path + 'test/')

    pickle.dump(C, open(path + "test/test_constants.pkl", "wb"))
    pickle.dump(sol, open(path + "test/test_data.pkl", "wb"))
    return 1


if __name__ == "__main__":
    for kx in C.K1_TRAIN:
        for ky in C.K2_TRAIN:
            generate_basis(kx, ky, 'train')

    create_train_data(options="lt")
