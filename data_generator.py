import math
import pickle
import time
import os


import numpy as np


from constants import Constants
from utils import f_a

path = Constants.PATH


def generate_train_data(k1_train, k2_train):
    Ex = []
    Ey1 = []
    Ey2 = []
    Hx_x = []
    Hx_y1 = []
    Hx_y2 = []
    Hy_x = []
    Hy_y1 = []
    Hy_y2 = []
    Energy = []

    for k1 in k1_train:
        for k2 in k2_train:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
            t = np.linspace(0, Constants.T, Constants.TIME_STEPS)

            fe = np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) + \
                 np.sin(Constants.PI * k2 * Constants.X) * np.sin(
                Constants.PI * k1 * Constants.Y)
            ex = [a * fe for a in np.cos(c * t)]
            ey1 = [a * fe for a in np.cos(c * (t + Constants.DT))]
            ey2 = [a * fe for a in np.cos(c * (t + 2 * Constants.DT))]

            fhx = (1 / c) * (-Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
                Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
                Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

            hx_x = [a * fhx[1:-1, :-1] for a in np.sin(c * (t + Constants.DT / 2))]
            hx_y1 = [a * fhx[1:-1, :-1] for a in np.sin(c * (t + 3 * Constants.DT / 2))]
            hx_y2 = [a * fhx[1:-1, :-1] for a in np.sin(c * (t + 5 * Constants.DT / 2))]

            fhy = (1 / c) * (
                    Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
                Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
                Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

            hy_x = [a * fhy[:-1, 1:-1] for a in np.sin(c * (t + Constants.DT / 2))]
            hy_y1 = [a * fhy[:-1, 1:-1] for a in np.sin(c * (t + 3 * Constants.DT / 2))]
            hy_y2 = [a * fhy[:-1, 1:-1] for a in np.sin(c * (t + 5 * Constants.DT / 2))]

            if k1 == k2:
                energy = [1] * Constants.TIME_STEPS
            else:
                energy = [1 / 2] * Constants.TIME_STEPS

            Ex.append(np.vstack(ex))
            Ey1.append(np.vstack(ey1))
            Ey2.append(np.vstack(ey2))
            Hx_x.append(np.vstack(hx_x))
            Hx_y1.append(np.vstack(hx_y1))
            Hx_y2.append(np.vstack(hx_y2))
            Hy_x.append(np.vstack(hy_x))
            Hy_y1.append(np.vstack(hy_y1))
            Hy_y2.append(np.vstack(hy_y2))
            Energy.append(np.vstack(energy))
    print('saving files')
    start_time = time.time()


    pickle.dump(Ex, open(path + "Ex_train.pkl", "wb"))
    pickle.dump(Ey1, open(path + "Ey1_train.pkl", "wb"))
    pickle.dump(Ey2, open(path + "Ey2_train.pkl", "wb"))
    pickle.dump(Hx_x, open(path + "Hx_x_train.pkl", "wb"))
    pickle.dump(Hx_y1, open(path + "Hx_y1_train.pkl", "wb"))
    pickle.dump(Hx_y2, open(path + "Hx_y2_train.pkl", "wb"))
    pickle.dump(Hy_x, open(path + "Hy_x_train.pkl", "wb"))
    pickle.dump(Hy_y1, open(path + "Hy_y1_train.pkl", "wb"))
    pickle.dump(Hy_y2, open(path + "Hy_y2_train.pkl", "wb"))
    pickle.dump(Energy, open(path + "Energy_train.pkl", "wb"))

    print("--- %s seconds ---" % (time.time() - start_time))
    return 1


def create_train_data(options='all'):
    with open(path + 'Ex_train.pkl', 'rb') as file:
        Ex_train = pickle.load(file)
    with open(path + 'Ey1_train.pkl', 'rb') as file:
        Ey1_train = pickle.load(file)
    with open(path + 'Ey2_train.pkl', 'rb') as file:
        Ey2_train = pickle.load(file)

    with open(path + 'Hx_x_train.pkl', 'rb') as file:
        Hx_x_train = pickle.load(file)
    with open(path + 'Hx_y1_train.pkl', 'rb') as file:
        Hx_y1_train = pickle.load(file)
    with open(path + 'Hx_y2_train.pkl', 'rb') as file:
        Hx_y2_train = pickle.load(file)

    with open(path + 'Hy_x_train.pkl', 'rb') as file:
        Hy_x_train = pickle.load(file)
    with open(path + 'Hy_y1_train.pkl', 'rb') as file:
        Hy_y1_train = pickle.load(file)
    with open(path + 'Hy_y2_train.pkl', 'rb') as file:
        Hy_y2_train = pickle.load(file)

    with open(path + 'Energy_train.pkl', 'rb') as file:
        Energy_train = pickle.load(file)

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

    for i in range(Constants.TRAIN_NUM):
        if options=='all':
            a = abs(np.random.rand(len(Ex_train)))
            a = a / a.sum()

            ex = np.sum(np.array([a * b for a, b in zip(a, Ex_train)]), axis=0)
            ey1 = np.sum(np.array([a * b for a, b in zip(a, Ey1_train)]), axis=0)
            ey2 = np.sum(np.array([a * b for a, b in zip(a, Ey2_train)]), axis=0)

            hx_x = np.sum(np.array([a * b for a, b in zip(a, Hx_x_train)]), axis=0)
            hx_y1 = np.sum(np.array([a * b for a, b in zip(a, Hx_y1_train)]), axis=0)
            hx_y2 = np.sum(np.array([a * b for a, b in zip(a, Hx_y2_train)]), axis=0)

            hy_x = np.sum(np.array([a * b for a, b in zip(a, Hy_x_train)]), axis=0)
            hy_y1 = np.sum(np.array([a * b for a, b in zip(a, Hy_y1_train)]), axis=0)
            hy_y2 = np.sum(np.array([a * b for a, b in zip(a, Hy_y2_train)]), axis=0)

            energy = np.sum(np.array([a * b for a, b in zip(a, Energy_train)]), axis=0)
        else:
            ex = Ex_train[i]
            ey1 = Ey1_train[i]
            ey2 =Ey2_train[i]

            hx_x = Hx_x_train[i]
            hx_y1 = Hx_y1_train[i]
            hx_y2 = Hx_y2_train[i]

            hy_x = Hy_x_train[i]
            hy_y1 = Hy_y1_train[i]
            hy_y2 =Hy_y2_train[i]

            energy =  Energy_train[i]

        EX.append(ex)
        EY1.append(ey1)
        EY2.append(ey2)

        HX_X.append(hx_x)
        HX_Y1.append(hx_y1)
        HX_Y2.append(hx_y2)

        HY_X.append(hy_x)
        HY_Y1.append(hy_y1)
        HY_Y2.append(hy_y2)

        ENERGY.append(energy)

    isExist = os.path.exists(path+ 'train/')
    if not isExist:
        os.makedirs(path+ 'train/')

    pickle.dump(np.vstack(EX).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "train/ex.pkl", "wb"))
    pickle.dump(np.vstack(EY1).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "train/ey1.pkl", "wb"))
    pickle.dump(np.vstack(EY2).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "train/ey2.pkl", "wb"))

    pickle.dump(
        np.vstack(HX_X).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
        open(path + "train/hx_x.pkl", "wb"))
    pickle.dump(
        np.vstack(HX_Y1).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
        open(path + "train/hx_y1.pkl", "wb"))
    pickle.dump(
        np.vstack(HX_Y2).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
        open(path + "train/hx_y2.pkl", "wb"))

    pickle.dump(
        np.vstack(HY_X).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
        open(path + "train/hy_x.pkl", "wb"))
    pickle.dump(
        np.vstack(HY_Y1).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
        open(path + "train/hy_y1.pkl", "wb"))
    pickle.dump(
        np.vstack(HY_Y2).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
        open(path + "train/hy_y2.pkl", "wb"))

    pickle.dump(np.vstack(ENERGY).reshape((Constants.TRAIN_NUM * Constants.TIME_STEPS, 1)),
                open(path + "train/energy_y.pkl", "wb"))

    return 1


def generate_test_data(k1_test, k2_test):
    ex = []
    hx_x = []
    hy_x = []
    for k1 in k1_test:
        for k2 in k2_test:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))

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

    isExist = os.path.exists(path+ 'test/')
    if not isExist:
        os.makedirs(path+ 'test/')

    pickle.dump(ex.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N, Constants.N, 1)),
                open(path + "test/ex_test.pkl", "wb"))
    pickle.dump(hx_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 2, Constants.N - 1, 1)),
                open(path + "test/hx_x_test.pkl", "wb"))
    pickle.dump(hy_x.reshape((len(k1_test) * len(k2_test) * Constants.TIME_STEPS, Constants.N - 1, Constants.N - 2, 1)),
                open(path + "test/hy_x_test.pkl", "wb"))
    return 1

if __name__ == "__main__":
    generate_train_data(Constants.K1_TRAIN, Constants.K2_TRAIN)
