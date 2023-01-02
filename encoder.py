### General Imports ###
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import circulant, expm, eig
### Autoencoder ###


import tensorflow as tf

import numpy as np
from scipy.optimize import minimize

n = 11
x = np.linspace(0, 1, n + 1)[:-1]
dx = x[1] - x[0]

A0 = np.random.rand(int(n*(n-1)/2),)


def func(A):
    err = 0
    for k in [1,2,3,4,5]:
        f = np.sin(k * x)
        g = k * np.cos(k * x)
        err += np.mean(abs(create_anti(A, n) @ f - g) ** 2)
    return err/4


def create_anti(kernel, n):
    assert(2*len(kernel))==n*(n-1)
    D=np.zeros((n,n))
    ind=0
    for i in range(n):
        for j in range(n):
            if i>j:
                D[i,j]=kernel[ind]
                D[j,i]=-kernel[ind]
                ind+=1
    return D


res = minimize(func, A0, method='BFGS', options={'disp': True}, tol=1e-9)
# print(res['x'])
# print(res['fun'])

print(q)

a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
print(eig(a))
print(q)

# from tensorflow.keras import callbacks
# from tensorflow.keras import regularizers
N = 10
x = np.linspace(0, 2 * math.pi, N + 1)[:-1]
dx = x[1] - x[0]
t = np.linspace(0, 1, 10)
dt = t[1] - t[0]
k = 3
f = np.sin(x)
kernely = np.zeros((N, 1))
kernely[-1] = 1
kernely[0] = 0
kernely[1] = -1

Dy = circulant(kernely) / 2

# Dy[0,-1]=0
# Dy[-1,0]=0

# Dy[0,-1]=0
# Dy[-1,0]=0
u0 = f
for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    print(np.max(abs(u0 - np.sin(x + n * dt))))
    sin = (expm(1j * Dy / dx) + expm(-1j * Dy / dx)) / (2 * 1j)
    u0 = u0 + dt * np.matmul(Dy, u0)
# Dy[0,-1]=0
# Dy[-1,0]=0

print(q)

from tensorflow.python.keras import backend as K

import scipy.optimize as scop
from scipy.interpolate import interp1d

# from tensorflow.keras.models import Model, model_from_json
#
# from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, \
#     MaxPooling2D, UpSampling2D, Input
#
# from tensorflow.keras.datasets import mnist
x = np.linspace(0, math.pi, 5)[1:-1]
dt = 1 / 20
h = x[1] - x[0]
f = np.sin(x)
lf = np.sin(x)
D1 = -np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])
print(np.matmul(D1, D1))
D2 = -np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
D, P = np.linalg.eig(D2)

Sd2 = np.matmul(P, np.matmul(np.diag(np.sqrt(D)), np.transpose(P)))
print(Sd2)
# print(np.matmul(Sd2, f)-np.cos(x))
print(q)

# D, P=np.linalg.eig(-1j*D1)
#
#
# u,s,v=np.linalg.svd(-1j*D1,compute_uv=True)
D2 = np.matmul(P, np.matmul(np.diag(np.exp(1j * D * dt)), np.transpose(P)))
# u=f+(dt/h)*np.matmul(D,f)
u0 = f
for j in range(2):
    # u0 = u0 + dt * np.matmul(D1, u0)
    u0 = np.real(np.matmul(D2, u0))
    print(u0 - np.sin(dt * (j + 1) + x))
# print(np.matmul(D,f)-lf)

print(q)

print(q)


# k=8
# n=16
# xb=np.linspace(0,math.pi,k+1).reshape(k+1,1)
# xx=np.linspace(0,math.pi,n+1).reshape(n+1,1)
#
#
# x=np.sin(7*xx)
# b = np.sin(7 * xb)
#
# # btilde=np.random.rand(len(x),1)
# # btilde[::int(n/k)]=b
# # btilde=x
# # print(btilde)
# # print(b)
# # print(ee)
#
#
# E=np.zeros((k+1,n+1))
# for j in range(k+1):
#     E[j, int(n/k) * j] = 1
# kernel=np.zeros((n+1,1))
# kernel[-1]=1
# kernel[0]=-1
# L=circulant(kernel)
# L[-1,0]=0
# # print(L)
# # print(np.linalg.matrix_rank(L))
# # print(rr)
#
#
#
# # x_hat=np.matmul(D_inv,la*np.matmul(np.matmul(L.T,L),btilde)+np.matmul(E.T,b))
# def func(la,*args):
#     D = la * np.matmul(L.T, L) + np.matmul(E.T, E)
#     u, s, vt = np.linalg.svd(D, full_matrices=False, compute_uv=True)
#     D_inv = np.matmul(np.matmul(vt.T, np.diag(1 / s)), u.T)
#     x_hat = np.matmul(D_inv, np.matmul(E.T, b))
#     # x_hat[::int(n / k)] = b
#
#     loss=np.sum((np.matmul(E,x_hat)-b)**2)+0.01*np.sum(np.matmul(L,(x_hat))**2)
#     return loss
#
#
# init =1/12.878
# # res = scop.minimize(func, init, method='SLSQP',
# #                     options=dict(disp=False, iprint=2, ftol=1e-8))
# # la=res['x']
# # print(la)
# # print(res['fun'])
# D = la * np.matmul(L.T, L) + np.matmul(E.T, E)
# u, s, vt = np.linalg.svd(D, full_matrices=False, compute_uv=True)
# D_inv = np.matmul(np.matmul(vt.T, np.diag(1 / s)), u.T)
# x_hat = np.matmul(D_inv, np.matmul(E.T, b))
# # x_hat[::int(n/k)]=b
# # x_hat= np.matmul(E.T, b)
# plt.plot(xx,x_hat, 'r')
# #
# # f2 = interp1d(np.squeeze(xx), np.squeeze(x_hat), kind='cubic')
# plt.plot(xx,x)
# # t=np.linspace(0,math.pi,200)
# # plt.plot(t,f2(t),'-r')
# plt.scatter(xb,b)
# # # plt.plot(b,'b')
# plt.show()
#
#
#
# print(qq)
#
#
# x0=x
# f0=np.sin(10*x)
# f=f0
# index=2
# p=x[index]
# fp=f0[index]
# h=1/n
#
# a=np.random.normal(0.,2,2)
# a=[0.1]
# kernel = np.array([0.5 + a[0], -a[0]] + [0 for i in range(len(f) - 4)] + [-a[0], 0.5 + a[0]])
# A = circulant(kernel)
# A_new = np.zeros(((2 * len(f)), len(f)))
# A_new[1::2] = A
# A_new[::2] = np.eye(len(f))
# f_new = np.matmul(A_new, f)
# x_new=np.linspace(p-1/(len(f_new)),p+1/(len(f_new)),len(f_new)+1)[:-1]
#
#
# # x_new = np.linspace(0, math.pi, len(f_new)+1)[:-1]
# # plt.plot(x_new,f_new)
# # plt.plot(x_new,np.sin(2*x_new))
# # plt.scatter(p,f_new[4])
#
# a=[0.1]*5
# for i, k in enumerate(range(len(a))):
#     kernel = np.array([0.5 + a[k], -a[k]] + [0 for i in range(len(f) - 4)] + [-a[k], 0.5 + a[k]])
#     A = circulant(kernel)
#     A_new = np.zeros(((2 * len(f)), len(f)))
#     A_new[1::2] = A
#     A_new[::2] = np.eye(len(f))
#
#     f_new = np.matmul(A_new, f)
#     # x_new = np.linspace(p - 1 / (len(f_new)), p + 1 / (len(f_new)), len(f_new) + 1)[:-1]
#     # print(x_new[int(len(f_new)/2)]-p)
#     x_new = np.linspace(0, math.pi, len(f_new)+1)[:-1]
#     print(x_new[2**(i+index)]-p)
#     ind=np.argmin(abs(x_new-p))
#     f=f_new
#
#
#
# # ind=[ind-3:ind+3]
# plt.plot(x_new,f_new)
# plt.plot(x_new,np.sin(10*x_new))
# plt.scatter(p,np.sin(10*p))
# plt.scatter(x0,f0)
# plt.show()
#
#
#
#
#
# print(q)
# kernel=np.zeros((n+1,1))
# kernel[-1]=1
# kernel[0]=-1
# L=circulant(kernel)
# L[-1,0]=0

# kernel=np.zeros((n+1,1))
# kernel[-1]=1
# kernel[0]=-1
# L=circulant(kernel)
# L[-1,0]=0


# f_test=np.zeros([len(k2)*2,n,1])
# f_tag_test=np.zeros([len(k2)*2,n,1])
def SubPixel1D(I, r):
    X = tf.transpose(a=I, perm=[2, 1, 0])  # (r, w, b)

    X = tf.batch_to_space(X, [r], [[0, 0]])  # (1, r*w, b)
    X = tf.transpose(a=X, perm=[2, 1, 0])
    return X


earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=50,
                                        restore_best_weights=False)

checkpoint_filepath = 'ed_weights'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
# csv loger
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=1e-9)


def custom_loss(y_true, y_pred):
    return tf.reduce_mean((abs(y_true - y_pred)) ** 2)


def reg_loss(f, A):
    return np.sqrt(np.sum(np.matmul(A, f) ** 2))


n = 40
n_sub = 20
x = np.linspace(0, math.pi, n + 1)[:-1].reshape(n, 1)
k1 = np.arange(35, 38, 1)

f_train = np.zeros([len(k1), n, 1])
f_tag_train = np.zeros([len(k1), n, 1])

sin = [np.sin(k * x) for k in k1]
cos = [np.cos(k * x) for k in k1]

for i, j in enumerate(k1):
    f_train[i] = sin[i]

    f_tag_train[i] = j * cos[i]

f_train = (f_train + 1) / 2

f_train_sub = f_train[:, ::int(n / n_sub)]
f_tag_train_sub = f_tag_train[:, ::int(n / n_sub)]

# print(tensorflow.reduce_mean(abs(tensorflow.reduce_sum(f_train**2,2)/f_tag[:,:,:,0]-1)))
# print(tt)
# for i,k in enumerate(k2):
#     f_test[i]=(np.sin(k*x))/n
#     y=(k*np.cos(k*x))/n
#     f_tag_test[i]=y

shape_x = n
input_dim = shape_x
input_img = Input(shape=(shape_x, 1))

# subsampling
X = tf.keras.layers.Dense(input_dim, activation='relu')(input_img)
X = tf.keras.layers.Conv1D(5, 5, activation='relu', padding="same", kernel_regularizer=regularizers.L2(0.01))(X)

X = tf.keras.layers.Dense(5, activation='relu')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Conv1D(1, 3, strides=2, padding="same", name='subsampled')(X)
Y = X
X = tf.keras.layers.Dense(18, activation='relu', kernel_regularizer=regularizers.L2(0.01), name='decode1')(X)
X = tf.keras.layers.BatchNormalization()(X)

X = tf.transpose(a=X, perm=[2, 1, 0])
X = tf.batch_to_space(X, [2], [[0, 0]])  # (1, r*w, b)
X = tf.transpose(a=X, perm=[2, 1, 0])

X = tf.keras.layers.Conv1D(5, 3, activation='relu', padding="same", name='decode3',
                           kernel_regularizer=regularizers.L2(0.01))(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(5, activation='relu', name='decode4', kernel_regularizer=regularizers.L2(0.01))(X)
X = tf.keras.layers.Conv1D(1, 5, name='decode5', activation='relu', padding="same",
                           kernel_regularizer=regularizers.L2(0.01))(X)
# X = tf.keras.layers.Dense(1, name='decode6',kernel_regularizer=regularizers.L2(0.01))(X)
# X=tf.keras.layers.BatchNormalization()(X)


X = tf.keras.layers.Dense(1, name='decode7')(X)

autoencoder = Model(inputs=[input_img], outputs=[X, Y])
autoencoder2 = Model(inputs=[Y], outputs=[Y])
print(q)

autoencoder.compile(optimizer='adam', loss=['mse', 'mse'], loss_weights=[1, 0.5])
# autoencoder.summary()
# print(q)

# history = autoencoder.fit(
#     [f_train], [f_train, f_train_sub],
#     callbacks=[model_checkpoint_callback, reduce_lr],
#     epochs=600,
#     batch_size=1,
#     shuffle=True, validation_split=0.1, verbose=2)
#
# plt.plot(history.history['loss'], "r", label="Loss")
# plt.plot(history.history['val_loss'], "b", label="Validation")
# plt.show()
# print(qq)

autoencoder.load_weights('ed_weights')
y3 = autoencoder.predict(f_train)[0]

y2 = f_train
#
# for i in np.arange(0, 6, 1):
#     print(autoencoder.layers[i].name)
#     y2 = autoencoder.layers[i](y2)


# y1 = f_train_sub[1]

y2 = y2[1]
y3 = y3[1]

plt.plot(y2)
plt.plot(y3)
plt.show()
print(q)

print(np.mean(autoencoder.predict(f_test)[1] - f_tag_test) ** 2)
print(rr)
# print(autoencoder.trainable_weights)
h = math.pi / n
kernel = np.zeros((n, 1))
kernel[0] = -1
kernel[-1] = 1
A = circulant(kernel / h)

# u,s,v= np.linalg.svd(decoded_imgs[i],full_matrices=False, compute_uv=True)
# print(reg_loss(f_test[i, :, :, 0].T, A))
# print(s[0])
# print(np.sqrt(f_tag_test[i]))
# print(q)
# i=2
# print(decoded_imgs[i].shape)
# print(f_tag[i,0,0,0]-tensorflow.reduce_sum(decoded_imgs[i]**2))
# # print(history.history['loss'])
# print(custom_loss(f_tag,decoded_imgs))


plt.show()
