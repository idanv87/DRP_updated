import pickle
import time
import os
import tracemalloc

import numpy as np

from constants import Constants
from utils import fE, fHX, fHY

C = Constants()
path = C.PATH


def generate_basis(k1, k2, name):
    assert name in ['train', 'test']
    l = dict()

    if k1 == k2:
        Energy = [1] * C.TIME_STEPS
    else:
        Energy = [1 / 2] * C.TIME_STEPS

    t, x, y = np.meshgrid(np.linspace(0, C.T, C.TIME_STEPS), C.X1, C.X2, indexing='ij')

    c = C.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

    FE = np.sin(C.PI * k1 * x) * np.sin(C.PI * k2 * y) + \
         np.sin(C.PI * k2 * x) * np.sin(C.PI * k1 * y)
    [l.setdefault(key, value) for key, value in
     zip(['ex', 'ey1', 'ey2'], [fE(FE, 0, t, c), fE(FE, 1, t, c), fE(FE, 2, t, c)])]
    del FE

    FHX = -C.PI * k2 * np.sin(C.PI * k1 * x) * np.cos(
        C.PI * k2 * (y + C.DX / 2)) - C.PI * k1 * np.sin(
        C.PI * k2 * x) * np.cos(
        C.PI * k1 * (y + C.DX / 2))
    [l.setdefault(key, value) for key, value in
     zip(['hx_x', 'hx_y1', 'hx_y2'], [fHX(FHX, 1, t, c), fHX(FHX, 3, t, c), fHX(FHX, 5, t, c)])]
    del FHX

    FHY = C.PI * k1 * np.cos(C.PI * k1 * (x + C.DX / 2)) * np.sin(
        C.PI * k2 * y) + C.PI * k2 * np.cos(
        C.PI * k2 * (x + C.DX / 2)) * np.sin(
        C.PI * k1 * y)
    [l.setdefault(key, value) for key, value in
     zip(['hy_x', 'hy_y1', 'hy_y2', 'energy'],
         [fHY(FHY, 1, t, c), fHY(FHY, 3, t, c), fHY(FHY, 5, t, c), np.vstack(Energy)])
     ]
    del FHY

    saving_path=path + 'base_functions/'+name+'/'
    isExist = os.path.exists(saving_path)
    if not isExist:
        os.makedirs(saving_path)

    pickle.dump(l, open(saving_path + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl', "wb"))

    return 1


def create_train_data(options='lt'):
    sol = {'ex': [], 'ey1': [], 'ey2': [], 'hx_x': [], 'hx_y1': [], 'hx_y2': [], 'hy_x': [], 'hy_y1': [],
           'hy_y2': [],
           'energy': []}

    if options == 'lt':

        for i in np.arange(C.TRAIN_NUM):

            dict1 = {'ex': 0, 'ey1': 0, 'ey2': 0, 'hx_x': 0, 'hx_y1': 0, 'hx_y2': 0, 'hy_x': 0, 'hy_y1': 0,
                    'hy_y2': 0,
                    'energy': 0}

            for k1 in C.K1_TRAIN:
                for k2 in C.K2_TRAIN:
                    with open(path + 'base_functions/train/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl',
                              'rb') as file:
                        l = pickle.load(file)

                    a = np.random.rand(1)/(len(C.K1_TRAIN)*len(C.K2_TRAIN))
                    [dict1.__setitem__(name, dict1[name] + a * l[name]) for name in list(dict1)]

            [sol[name].append(dict1[name]) for name in list(dict1)]


    else:
        for k1 in C.K1_TRAIN:
            for k2 in C.K2_TRAIN:
                with open(path + 'base_functions/train/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl',
                          'rb') as file:
                    l = pickle.load(file)
                [sol[name].append(l[name]) for name in list(sol)]

    isExist = os.path.exists(path + 'train/')
    if not isExist:
        os.makedirs(path + 'train/')

    pickle.dump(C, open(path + "train/train_constants.pkl", "wb"))
    pickle.dump(sol, open(path + "train/train_data.pkl", "wb"))

    return 1


def create_test_data(k1_test=C.K1_TEST, k2_test=C.K2_TEST):
    for kx in k1_test:
        for ky in k2_test:
            generate_basis(kx, ky,'test')

    sol = {'ex': [], 'hx_x': [], 'hy_x': []}
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
            generate_basis(kx, ky,'train')

    create_train_data(options="lt")
