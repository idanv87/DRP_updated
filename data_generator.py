import pickle
import time
import os
import tracemalloc
import math

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import numpy as np

from DRP_multiple_networks.constants import Constants
from DRP_multiple_networks.auxilary.aux_functions import fE, fHX, fHY, dim_red1, dim_red2, chebyshev_nodes, compute_Q

"""
This file is used to gennerate data for training and for evaluation
"""

C = Constants()

path = C.PATH

folders = [path + 'train/', path + 'test/', path + 'val/', path + 'base_functions/train/',
           path + ' base_functions/val/',
           path + 'base_functions/test/', path + 'figures/']

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


class base_function:
    """
    an instance of  base_function is an analytic solution (e,hx,hy) to maxwell equations.
    each solution has energy.
    each solution can be used either for train or either for test
     """
    base_pathes = {'train': [], 'test': []}

    def __init__(self, k1, k2, train_or_test):
        assert train_or_test in list(base_function.base_pathes)
        self.path = path + 'base_functions/' + train_or_test + '/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl'
        base_function.base_pathes[train_or_test].append(self.path)
        self.valueFE = None
        self.valueFHX = None
        self.valueFHY = None
        self.valueENERGY = None

    def set(self, typ, F):
        if typ == 'e':
            self.valueFE = F
        if typ == 'hx':
            self.valueFHX = F
        if typ == 'hy':
            self.valueFHY = F
        if typ == 'energy':
            self.valueENERGY = F

    def save(self):
        assert [self.valueFE, self.valueFHX, self.valueFHY, self.valueENERGY] not in [None]
        pickle.dump({'e': self.valueFE, 'hx': self.valueFHX, 'hy': self.valueFHY, 'energy': self.valueENERGY},
                    open(self.path, "wb"))


def create_lt(name, ind):
    """
    please ignore that class for now
    """
    output = {'e': 0., 'hx': 0., 'hy': 0., 'energy': 0.}

    if ind == 0:
        for p in list(base_function.base_pathes[name]):

            with open(p, 'rb') as file:
                l = pickle.load(file)
            for key in list(output):
                output[key] += l[key]
    else:
        a = np.zeros(len(base_function.base_pathes[name]))
        a[ind + 15] = 1
        for i, p in enumerate(list(base_function.base_pathes[name])):
            # a=np.random.rand(1)
            with open(p, 'rb') as file:
                l = pickle.load(file)
            for key in list(output):
                output[key] += a[i] * l[key]

    return output


def create_train_data(gen_base, options):
    """
    let f=(e,hx,hy).
    this function creates 2 lists   of  inputs (fn,fn+1,fn+2) and outputs (fn+1,fn+2,fn+3)
     for the network and save them in 2 different files (so list of 9 elements each
     each element contains the number of base functions times the time steps-3
     ).
    the data can be generated according options: either by a single base function or either as a random
     linear transformation (options=lt) of base functions

    """

    sol = {'e': [], 'hx': [], 'hy': [], 'energy': []}
    generate_basis('train')

    if options == 'lt':
        for i in range(C.TRAIN_NUM):
            [sol[key].append(create_lt('train', i)[key].copy()) for key in list(sol)]
    else:
        for p in list(base_function.base_pathes['train']):
            with open(p, 'rb') as file:
                l = pickle.load(file)
            [sol[key].append(l[key].copy()) for key in list(sol)]

    # s['e'] is a list of all test functions.
    net_input = dim_red2(sol, 0) + dim_red2(sol, 1) + dim_red2(sol, 2)
    net_output = dim_red2(sol, 2) + dim_red1(sol, 3)

    pickle.dump(net_input, open(path + 'train/input.pkl', "wb"))
    pickle.dump(net_output, open(path + 'train/output.pkl', "wb"))

    return 1


def generate_basis(name, h=Constants.DX, dt=Constants.DT, t_f=Constants.T, time_steps=Constants.TIME_STEPS,
                   X1=Constants.X1, X2=Constants.X2):
    """
    This function generate the base functions and their energy given by the modes in the file constants.k_train
    """
    assert name in ['train', 'test']
    if name == 'train':
        kx = C.K1_TRAIN
        ky = C.K2_TRAIN
    else:
        kx = C.K1_TEST
        ky = C.K2_TEST
    t, x, y = np.meshgrid(np.linspace(0, t_f, time_steps), X1, X2, indexing='ij')
    for k1 in kx:
        for k2 in ky:
            B = base_function(k1, k2, name)

            c = C.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

            B.set('e', fE(t, x, y, k1, k2, c))

            B.set('hx', fHX(t, x, y, k1, k2, c, h))

            B.set('hy', fHY(t, x, y, k1, k2, c, h))
            if k1 == k2:
                B.set('energy', np.vstack([1.] * time_steps))
            else:
                B.set('energy', np.vstack([1 / 2] * time_steps))

            B.save()

    return 1


def create_test_data(options='lt', h=Constants.DX, dt=Constants.DT, t_f=Constants.T, time_steps=Constants.TIME_STEPS,
                     X1=Constants.X1, X2=Constants.X2):
    """
    test_data[e] is a list and each element is a base function of rank 3-(t,x,y)
    for test.
    it can be l_t if one wants.
    """

    L1 = []
    L2 = []
    L3 = []
    generate_basis('test', h, dt, t_f, time_steps, X1, X2)
    for p in list(base_function.base_pathes['test']):
        with open(p, 'rb') as file:
            l = pickle.load(file)
        L1.append(l['e'])
        L2.append(l['hx'])
        L3.append(l['hy'])

    test_data = {'e': L1, 'hx': L2, 'hy': L3}
    if options == 'lt':
        test_data['e'] = [sum(test_data['e'])]
        test_data['hx'] = [sum(test_data['hx'])]
        test_data['hy'] = [sum(test_data['hy'])]

    pickle.dump(test_data, open(path + 'test/test_data.pkl', "wb"))

    return 1


if __name__ == "__main__":
    n = 10
    t_steps =10
    dt = 1 / (t_steps - 1)
    # e=np.random.uniform(0, math.pi, size=(n, 2))
    X=np.linspace(0,math.pi,n+1)[0:-1]
    Y=X
    t, x, y = np.meshgrid(np.linspace(0, 1, t_steps), X, Y,
                          indexing='ij')
    k1=[3,5,7]
    k2=[3,5,7]
    Q=compute_Q(t, x, y, k1, k2, t_steps, n)

    # Q_new=block_diag(Q, Q,Q,Q)
    Q_new=Q
    # Q_new=block_diag(Q_new, Q_new)

    k1=6
    k2=7


    # e=np.random.uniform(0, math.pi, size=(n, 2))
    X=np.linspace(0,math.pi,n+1)[0:-1]
    Y=X
    t_steps = 10
    t, x, y = np.meshgrid(np.linspace(0, 1, t_steps), X, Y,
                          indexing='ij')

    E = fE(t, x, y, k1, k2).real
    Hx = fHX(t, x, y, k1, k2).real
    Hy = fHY(t, x, y, k1, k2).real
    A = np.zeros((n * n * 3, t_steps - 1))
    B = np.zeros((n * n * 3, t_steps - 1))
    for ind in range(t_steps - 1):
            A[:, ind] = np.concatenate((E[ind].flatten(), Hx[ind].flatten(), Hy[ind].flatten()))
            B[:, ind] = np.concatenate((E[ind + 1].flatten(), Hx[ind + 1].flatten(), Hy[ind + 1].flatten()))

    print(A.shape)
    err=[np.mean((np.linalg.matrix_power(Q_new,k)@A[:,0]-B[:,k-1])**2) for k  in range(t_steps-1)][1:]
    print(np.mean(err))
    print(err)