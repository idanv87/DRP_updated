import pickle
import time
import os
import tracemalloc

import numpy as np

from DRP_multiple_networks.constants import Constants
# from DRP_multiple_networks.tests import train_constants, test_constants
from DRP_multiple_networks.auxilary.aux_functions import fE, fHX, fHY, dim_red1, dim_red2

"""
This file is used to gennerate data for training and for evaluation
"""

path = Constants.PATH

folders = [path + 'train/', path + 'test/', path + 'val/', path + 'base_functions/train/',
           path + ' base_functions/val/',
           path + 'base_functions/test/', path + 'figures/']

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


class base_function_test:
    """
    an instance of  base_function is an analytic solution (e,hx,hy) to maxwell equations.
    each solution has energy.
    each solution can be used either for train or either for test
     """

    def __init__(self, k1, k2, train_or_test):
        self.base_pathes = {'train': [], 'test': []}
        assert train_or_test in list(base_function.base_pathes)
        self.path = path + 'base_functions/' + train_or_test + '/' + 'kx=' + str(k1) + 'ky=' + str(k2) + '_.pkl'
        self.base_pathes[train_or_test].append(self.path)
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





def create_train_data(c, options='non_lt'):
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
    generate_basis('train', c)

    if options == 'lt':
       pass
    else:
        for p in list(base_function.base_pathes['train']):
            with open(p, 'rb') as file:
                l = pickle.load(file)
            [sol[key].append(l[key].copy()) for key in list(sol)]

    net_input = dim_red2(sol, 0) + dim_red2(sol, 1) + dim_red2(sol, 2)
    net_output = dim_red2(sol, 1) + dim_red2(sol, 2) + dim_red1(sol, 3)

    pickle.dump(net_input, open(path + 'train/input.pkl', "wb"))
    pickle.dump(net_output, open(path + 'train/output.pkl', "wb"))

    return 1


def generate_basis(name, c):
    """
    This function generate the base functions and their energy given by the modes in the file constants.k_train
    """
    if name == 'test':
        kx = c.K1_TEST
        ky = c.K2_TEST
    else:
        kx = c.K1_TRAIN
        ky = c.K2_TRAIN
    t, x, y = np.meshgrid(np.linspace(0, c.T, c.TIME_STEPS), c.X1, c.X2, indexing='ij')

    for k1 in kx:
        for k2 in ky:
            if name == 'test':
                B = base_function_test(k1, k2, name)
            else:
                B = base_function(k1, k2, name)

            omega = Constants.PI * (np.sqrt(k1 ** 2 + k2 ** 2))

            B.set('e', fE(t, x, y, k1, k2, omega))

            B.set('hx', fHX(t , x, y, k1, k2, omega, c.DX))

            B.set('hy', fHY(t , x, y, k1, k2, omega, c.DX))
            if k1 == k2:
                B.set('energy', np.vstack([1.] * c.TIME_STEPS))
            else:
                B.set('energy', np.vstack([1 / 2] * c.TIME_STEPS))

            B.save()

    return B


def create_test_data(c, options='lt'):
    """
    test_data[e] is a list and each element is a base function of rank 3-(t,x,y)
    for test.
    it can be l_t if one wants.
    """

    L1 = []
    L2 = []
    L3 = []
    B = generate_basis('test', c)
    for p in list(B.base_pathes['test']):
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

# if __name__ == "__main__":
#     create_train_data(test_constants)
