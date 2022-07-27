import math
import pickle
import time
import os

from constants import Constants
from utils import *


path = Constants.PATH

def generate_train_data(k1_train, k2_train):
    Energy = []


    for kx in k1_train:
        for ky in k2_train:

            if kx == ky:
                energy = [1] * Constants.TIME_STEPS
            else:
                energy = [1 / 2] * Constants.TIME_STEPS

            Energy.append(np.vstack(energy))

    FE=np.sin(Constants.PI * Constants.k1 * Constants.x) * np.sin(Constants.PI * Constants.k2 * Constants.y)+ \
       np.sin(Constants.PI * Constants.k2 * Constants.x) * np.sin(Constants.PI * Constants.k1 * Constants.y)

    FHX=-Constants.PI * Constants.k2 * np.sin(Constants.PI * Constants.k1 * Constants.x) * np.cos(
        Constants.PI * Constants.k2 * (Constants.y + Constants.DX / 2)) - Constants.PI * Constants.k1 * np.sin(
        Constants.PI * Constants.k2 * Constants.x) * np.cos(Constants.PI * Constants.k1 * (Constants.y + Constants.DX / 2))

    FHY=  Constants.PI * Constants.k1 * np.cos(Constants.PI * Constants.k1 * (Constants.x + Constants.DX / 2)) * np.sin(
        Constants.PI * Constants.k2 * Constants.y) + Constants.PI * Constants.k2 * np.cos(
        Constants.PI * Constants.k2 * (Constants.x + Constants.DX / 2)) * np.sin(Constants.PI * Constants.k1 * Constants.y)
    return fE(FE,0), fE(FE,1), fE(FE,2), fHX(FHX,1), fHX(FHX,3), fHX(FHX, 5), fHY(FHY, 1), fHY(FHY,3), fHY(FHY, 5), np.array(Energy)


def create_train_data(options='all'):
    EX = []
    EY1 = []
    EY2 = []
    HX_X = []
    HX_Y1 = []
    HX_Y2 = []
    HY_X = []
    HY_Y1 = []
    HY_Y2 = []
    ENERGY = []

    Ex_train, Ey1_train, Ey2_train, Hx_x_train, Hx_y1_train, Hx_y2_train, Hy_x_train, Hy_y1_train, Hy_y2_train, Energy_train = \
        generate_train_data(Constants.K1_TRAIN, Constants.K2_TRAIN)


    if options == 'all':

        for i in range(Constants.TRAIN_NUM):
            a = np.random.rand(len(Constants.K1_TRAIN) * len(Constants.K2_TRAIN))
            a = a / a.sum()

            EX.append(np.sum(np.array([a * Ex_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            EY1.append(np.sum(np.array([a * Ey1_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            EY2.append(np.sum(np.array([a * Ey2_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))

            HX_X.append(np.sum(np.array([a * Hx_x_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            HX_Y1.append(
                np.sum(np.array([a * Hx_y1_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            HX_Y2.append(
                np.sum(np.array([a * Hx_y2_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))

            HY_X.append(np.sum(np.array([a * Hy_x_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            HY_Y1.append(
                np.sum(np.array([a * Hy_y1_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))
            HY_Y2.append(
                np.sum(np.array([a * Hy_y2_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))

            ENERGY.append(
                np.sum(np.array([a * Energy_train[k] for a, k in zip(a, np.arange(Ex_train.shape[0]))]), axis=0))

            dim = np.vstack(EX).shape[0]

    else:
        dim = np.vstack(Ex_train).shape[0]
        EX = Ex_train
        EY1 = Ey1_train
        EY2 = Ey2_train
        HX_X = Hx_x_train
        HX_Y1 = Hx_y1_train
        HX_Y2 = Hx_y2_train
        HY_X = Hy_x_train
        HY_Y1 = Hy_y1_train
        HY_Y2 = Hy_y2_train
        ENERGY = Energy_train
    isExist = os.path.exists(path + 'train/')
    if not isExist:
        os.makedirs(path + 'train/')
    l=(np.vstack(EX).reshape((dim, Constants.N, Constants.N, 1)),
           np.vstack(EY1).reshape((dim, Constants.N, Constants.N, 1)),
           np.vstack(EY2).reshape((dim, Constants.N, Constants.N, 1)),
           np.vstack(HX_X).reshape((dim, Constants.N - 2, Constants.N - 1, 1)),
           np.vstack(HX_Y1).reshape((dim, Constants.N - 2, Constants.N - 1, 1)),
           np.vstack(HX_Y2).reshape((dim, Constants.N - 2, Constants.N - 1, 1)),
           np.vstack(HY_X).reshape((dim, Constants.N - 1, Constants.N - 2, 1)),
           np.vstack(HY_Y1).reshape((dim, Constants.N - 1, Constants.N - 2, 1)),
           np.vstack(HY_Y2).reshape((dim, Constants.N - 1, Constants.N - 2, 1)),
           np.vstack(ENERGY).reshape((dim, 1)))

    pickle.dump(l, open(path + 'train/train_data.pkl', "wb"))
    return 1


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
    create_train_data()
