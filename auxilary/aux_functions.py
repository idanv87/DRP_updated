
import numpy as np
import tensorflow as tf
import math

from DRP_multiple_networks.constants import Constants

C = Constants()


def relative_norm(A, B, p=2):
    return tf.math.reduce_mean(abs(A-B))


def fE(t, x, y, k1, k2):
    k=np.sqrt(k1**2+k2**2)
    return np.exp(1J*(k1*x+k2*y-k*t))



def fHX(t, x, y, k1, k2):
     k = np.sqrt(k1 ** 2 + k2 ** 2)
     return np.exp(1J*(k1*x+k2*y-k*t)*k2/k)


def fHY(t, x, y, k1, k2):
    k = np.sqrt(k1 ** 2 + k2 ** 2)
    return -np.exp(1J * (k1 * x + k2 * y - k * t) * k1 / k)


def dim_red1(dic, m):
    d = {key: dic[key].copy() for key in dic.keys()}
    for key in ['e', 'hx', 'hy', 'energy']:
        for i in range(len(d[key])):
            if m != 3:
                d[key][i] = d[key][i][m:m - 3]
            else:
                d[key][i] = d[key][i][m:]

    return list([np.expand_dims(np.vstack(d[key]), axis=-1) for key in ['e', 'hx', 'hy', 'energy']])


def dim_red2(dic, m):
    d = {key: dic[key].copy() for key in dic.keys()}
    for key in ['e', 'hx', 'hy']:
        for i in range(len(d[key])):
            if m != 3:
                d[key][i] = d[key][i][m:m - 3]
            else:
                d[key][i] = d[key][i][m:]

    return list([np.expand_dims(np.vstack(d[key]), axis=-1) for key in ['e', 'hx', 'hy']])

def chebyshev_nodes(a, b, n):
    # n Chebyshev noder i intervallet [a, b]
    i = np.array(range(n))
    x = np.cos((2*i+1)*math.pi/(2*(n))) # noder over intervallet [-1,1]
    return 0.5*(b-a)*x+0.5*(b+a) # noder over intervallet [a,b]

def compute_Q(t,x,y,kx,ky,t_steps,n):
    A_all=[]
    B_all=[]
    for k1 in kx:
      for k2 in ky:

         A = np.zeros((n * n * 3, (t_steps - 1)))
         B = np.zeros((n * n * 3, t_steps - 1))
         E = fE(t, x, y, k1, k2).real
         Hx = fHX(t, x, y, k1, k2).real
         Hy = fHY(t, x, y, k1, k2).real

         for ind in range(t_steps - 1):
            A[:, ind] = np.concatenate((E[ind].flatten(), Hx[ind].flatten(), Hy[ind].flatten()))
            B[:, ind] = np.concatenate((E[ind + 1].flatten(), Hx[ind + 1].flatten(), Hy[ind + 1].flatten()))
         A_all.append(A)
         B_all.append(B)

    u, s, v = np.linalg.svd(np.hstack(B_all) @ np.transpose(np.hstack(A_all)), compute_uv=True)
    # print(np.mean((u@v@A-B)**2))
    D=np.eye(u.shape[0])
    D[-1,-1]=np.linalg.det(u)*np.linalg.det(v)
    return D@u @ v