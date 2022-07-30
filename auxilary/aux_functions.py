import numpy as np

from DRP_multiple_networks.constants import Constants
C=Constants()

def fE(t, x,y,k1,k2, c):
    return np.cos(c * t)*(np.sin(C.PI * k1 * x) * np.sin(C.PI * k2 * y) + \
           np.sin(C.PI * k2 * x) * np.sin(C.PI * k1 * y))



def fHX(t, x,y,k1,k2, c):
    z = np.sin(c * t) * (1 / c) *(
       -C.PI * k2 * np.sin(C.PI * k1 * x) * np.cos(
        C.PI * k2 * (y + C.DX / 2)) - C.PI * k1 * np.sin(
        C.PI * k2 * x) * np.cos(
        C.PI * k1 * (y + C.DX / 2))
    )
    return z[:, 1:-1, :-1]


def fHY(t,x,y,k1,k2, c):

    z = np.sin(c * t) * (1 / c) * (
            C.PI * k1 * np.cos(C.PI * k1 * (x + C.DX / 2)) * np.sin(
        C.PI * k2 * y) + C.PI * k2 * np.cos(
        C.PI * k2 * (x + C.DX / 2)) * np.sin(
        C.PI * k1 * y)
        )
    return z[:, :-1, 1:-1]

def dim_red1(dic,m):
    d = {key: dic[key].copy() for key in dic.keys()}
    for key in ['e', 'hx', 'hy', 'energy']:
        for i in range(len(d[key])):
            if m!=3:
               d[key][i]=d[key][i][m:m-3]
            else:
                d[key][i] = d[key][i][m:]

    return list([ np.expand_dims(np.vstack(d[key]),axis=-1) for key in ['e', 'hx', 'hy', 'energy']])

def dim_red2(dic,m):
    d={key:dic[key].copy() for key in dic.keys() }
    for key in ['e', 'hx', 'hy']:
        for i in range(len(d[key])):
            if m!=3:
               d[key][i]=d[key][i][m:m-3]
            else:
                d[key][i] = d[key][i][m:]

    return list([np.expand_dims(np.vstack(d[key]),axis=-1) for key in ['e', 'hx', 'hy']])