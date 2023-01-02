import pickle
import tracemalloc
import math
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import polar
from scipy.stats import pearsonr

from DRP_multiple_networks.auxilary.aux_functions import relative_norm, frob_norm
from scipy.linalg import circulant
from scipy.sparse import diags
from DRP_multiple_networks.constants import Constants
t=np.linspace(0,(2*np.sqrt(2)/5),73)
n=7
x=np.linspace(0,1,n+1)

h=1/n
k1=3
k2=0

norm=1
f=np.sin(k1*2*math.pi*x)
ftag=(k1*math.pi*np.cos(k1*math.pi*x)+k2*math.pi*np.cos(k2*math.pi*x))/norm
ftagdiff=(k1*math.pi*np.cos(k1*math.pi*(x+h/2))+k2*math.pi*np.cos(k2*math.pi*(x+h/2)))/norm

plt.plot(np.fft.fft(f))
plt.show()

print(qq)
# print(np.sum(np.dot(F,np.conjugate(F)))/np.sum(f**2))
# print(rr)
kernel=np.zeros((n,1))
kernel[0]=-1
kernel[-1]=1
A=circulant(kernel/h)
print(A)
print(np.fft.fft(kernel.T))
print(rr)

for  k in range(len(f)):
    # F[k]=F[k] * 2 * math.pi * 1j * k
    if k<int(n/2):
        F[k]=F[k]*2*math.pi*1j*k
    if k>int(n/2):
       F[k]=F[k]*2*math.pi*1j*(k-n)
    if k==int(n/2):
        F[k]=0

print(frob_norm(np.real(np.fft.ifft(F)),0)-frob_norm(ftag,0))
print(frob_norm(np.matmul(A,f),0)-frob_norm(ftagdiff,0))
print(np.max(abs(np.matmul(A,f)-ftagdiff)))

plt.show()
print(rr)

#
# m1=2
# m2=4
# h=1/(20)
# # f=np.exp(1j*2*math.pi*m1*x)
# # g=np.exp(-1j*2*math.pi*m2*x)
# f=np.cos(math.pi*m1*x)
# g=np.cos(math.pi*m2*x)
#
# print(np.dot(f,g))
# print(rr)
T, X, Y = np.meshgrid(t, x, x, indexing='ij')
k1=1
k2=2
E=0
Hx=0
Hy=0
s=np.random.normal(0, 1,100)*0+1
# s=s/100
for i,k1 in enumerate(np.arange(2,3,1)):
    for k2 in np.arange(4,5,1):

       omega = Constants.PI * (np.sqrt((k1) ** 2 + k2 ** 2))

       E+=np.cos(omega * T) * np.sin(math.pi * k1 * X) * np.sin(math.pi * k2 * Y)

       Hx=Hx-np.sin(omega * T) * (1 / omega) *\
          (Constants.PI * k2 * np.sin(Constants.PI * k1 * X) * np.cos(
        Constants.PI * k2 * Y))
       Hy+=  np.sin(omega * T) * (1 / omega) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * X) * np.sin(
        Constants.PI * k2 * Y))

       # Hx=Hx[1:]-Hx[:-1]
       # Hy=Hy[1:]-Hy[:-1]

# E=np.array([E[i]/np.sqrt(frob_norm(E[0],0)) for i in range(len(t))])
# E=E[1:]-(E[:-1]+1e-14)

# Enew=np.array(tf.reshape(E[0], [E[0].shape[0]**2,1]))
# de, u01, v01 = tf.linalg.svd(Enew, full_matrices=False, compute_uv=True)


# de, u01, v01 = tf.linalg.svd(np.matmul(Enew,Enew.T), full_matrices=False, compute_uv=True)



# Hx=np.array([Hx[i]/np.sqrt(frob_norm(E[0],0)) for i in range(len(t))])
# Hy=np.array([E[i]/np.sqrt(frob_norm(E[0],0)) for i in range(len(t))])


se=[]
shx=[]
shy=[]
seq=[]
shxq=[]
shyq=[]
sE=[]

for i in range(len(t)-1):
   Enew=np.array(tf.reshape(E[i], [E[i].shape[0]**2,1]))
   Hxnew= np.array(tf.reshape(Hx[i], [Hx[i].shape[0]**2,1]))
   Hynew= np.array(tf.reshape(Hy[i], [Hy[i].shape[0]**2,1]))
   # de, u01, v01 = tf.linalg.svd(np.matmul(Enew,Enew.T), full_matrices=False, compute_uv=True)
   # dhx, u01, v01 = tf.linalg.svd(np.matmul(Hxnew, Hxnew.T), full_matrices=False, compute_uv=True)
   # dhy, u01, v01 = tf.linalg.svd(np.matmul(Hynew,Hynew.T), full_matrices=False, compute_uv=True)
   de, u02, v02 = tf.linalg.svd(Enew, full_matrices=False, compute_uv=True)
   dhx, u01, v01 = tf.linalg.svd(Hxnew, full_matrices=False, compute_uv=True)
   dhy, u01, v01 = tf.linalg.svd(Hynew, full_matrices=False, compute_uv=True)



   seq.append((abs(1-np.max(de))))
   shxq.append(((1-np.max(dhx))**2))
   shyq.append(((1-np.max(dhy))**2))
   # seq.append(frob_norm(Enew,0))
   # shxq.append(frob_norm(Hxnew,0))
   # shyq.append(frob_norm(Hynew,0))
   # seq.append(frob_norm(Enew,0)+frob_norm(Hxnew,0)+frob_norm(Hynew,0))
   seq.append(np.max(de)+np.max(dhx)+np.max(dhy))
   # se.append(np.max(de)**2)
   # shx.append((np.max(dhx))**2)
   # shy.append((np.max(dhy))**2)
# corr, _ = pearsonr(np.array(seq), E[1:,11,12])
# print(corr)
# corr, _ = pearsonr(np.array(shx), np.array(se))

plt.plot(np.array(seq))
# plt.plot(E[1:,11,12]**2)
plt.show()
print(rr)
# plt.plot(shx)
# plt.plot(shy)
# plt.plot(se)

# plt.plot([se[i]+shx[i]+shy[i] for i  in range(len(se))])
# print(corr)
plt.show()
print(qq)
# print(' saved as:dispersion_figure' + str(save[1]) + '.eps')
# plt.savefig(Constants.FIGURES_PATH + 'dispersion_figure' + save[1] + '.eps', format='eps',
#             bbox_inches='tight')



# a=np.arange(18).reshape(2,3,3,1)
# c=tf.reshape(a,[-1,a.shape[1]*a.shape[2],1])
# print(c.shape)
#
# b=tf.reshape(a,[2,9,1])
# print(c-b)
#
# print(q)
#
# n=21
# x=np.linspace(0,1,n)
# h=x[1]-x[0]
# k=17
# f=np.sin(k*math.pi*x).reshape(n,1)
# ftag=k*math.pi*np.cos(k*(x+h/2)*math.pi)[1:-2]
# i = tf.constant(f, dtype=tf.float32, name='i')
# data = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
#
# a = 9/8
# k = tf.constant([(a - 1) / 3, -a, a, (1 - a) / 3], dtype=tf.float32, name='k')
# kernel0 = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
# res0 = tf.squeeze((1 / h) * tf.nn.conv1d(data, kernel0, stride=1, padding='VALID'))
#
# s, u, v = tf.linalg.svd(tf.reshape(res0,[n-3,1]))
# print(s)
#
# print(q)
# s = (1/8)*np.random.normal(0, 1.2, 1000)+1
# err=[]
# nerr=[]
# for i in range(len(s)):
#    a=s[i]
#    k = tf.constant([(a-1)/3,-a,a,(1-a)/3], dtype=tf.float32, name='k')
#    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
#    res = tf.squeeze((1/h)*tf.nn.conv1d(data, kernel, stride=1, padding='VALID'))
#    err.append(tf.reduce_mean(abs(res-ftag)**2))
#    nerr.append(tf.reduce_mean(abs(res0 - ftag)**2))
# plt.plot(err)
# print(np.mean(err))
# print(np.mean(nerr))
# plt.plot(nerr)
# plt.plot(s)
# plt.show()
#
#
# print(q)
n=21
x=np.linspace(0,1,n)
h=x[1]-x[0]
k=11
k2=19
f=(3*np.sin(k*math.pi*x)+4*np.sin(k2*math.pi*x)).reshape(n,1)
f_tag=(3*math.pi*k*np.cos(k*math.pi*(x+h/2))+4*math.pi*k2*np.cos(k2*math.pi*(x+h/2)))[:-1].reshape(n-1,1)
ftag_red=(math.pi*k2*np.cos(k2*math.pi*(x+h/2)))[:-1].reshape(n-1,1)

g=((f[1:]-f[0:-1])/h).reshape(n-1,1)
A=np.matmul(f_tag,f_tag.T)
Ared=np.matmul(ftag_red,ftag_red.T)
B=np.matmul(g,g.T)


dared, u01, v01 = tf.linalg.svd(Ared, full_matrices=False, compute_uv=True)
da0, u0, v0 = tf.linalg.svd(A, full_matrices=False, compute_uv=True)
da1, u1, v1 = tf.linalg.svd(B, full_matrices=False, compute_uv=True)
# print(np.max(da0))
# print(rr)

ftag_g = tf.matmul(u1, tf.matmul(tf.linalg.diag(da1), v1, adjoint_b=True))
ftag_new = tf.matmul(u1, tf.matmul(tf.linalg.diag(da0), v1, adjoint_b=True))
# print(frob_norm(g,f_tag)/frob_norm(ftag_new,f_tag))

# print(np.max(da0)/np.max(da1))
# print(frob_norm(A,B)/frob_norm(A,ftag_new))
# print(frob_norm(A,ftag_new))



print(q)
p=polar(A)[1]
print(frob_norm(p,0))
# print(np.sum(f ** 2))
print(q)
# print(f)
A=np.matmul(f,f.T)
B=np.matmul(A.reshape( n**2,1),A.reshape( n**2,1).T)
C=np.matmul(B.reshape( n**4,1),B.reshape( n**4,1).T)
da0, u1, v1 = tf.linalg.svd(f, full_matrices=False, compute_uv=True)
da, u1, v1 = tf.linalg.svd(A, full_matrices=False, compute_uv=True)
db, u1, v1 = tf.linalg.svd(B, full_matrices=False, compute_uv=True)
dc, u1, v1 = tf.linalg.svd(C, full_matrices=False, compute_uv=True)
print(np.max(da0))
print(np.max(da))
print(np.max(db))
print(np.max(dc))
# print(np.linalg.matrix_rank(A))
print(q)
t=np.linspace(0,10,500)

T, X, Y = np.meshgrid(t, x, x, indexing='ij')
kx=17
ky=17
kx2=18
ky2=18
kx3=2
ky3=4
c3=math.pi*np.sqrt(kx3^2+ky3^2)
c2=math.pi*np.sqrt(kx2^2+ky2^2)
c=math.pi*np.sqrt(kx^2+ky^2)

u=np.cos(c*T)*np.sin(math.pi*kx*X)*np.sin(math.pi*ky*Y)
u2=np.cos(c2*T)*np.sin(math.pi*kx2*X)*np.sin(math.pi*ky2*Y)
x=np.linspace(0,1,20)
f=np.cos(6*x*math.pi).reshape(20,1)

d, u1, v1 = tf.linalg.svd(np.matmul(f,f.T), full_matrices=False, compute_uv=True)
d2, u1, v1 = tf.linalg.svd(f, full_matrices=False, compute_uv=True)
print(d)
print(d2)
print(q)


D=1-d
print(1-d[0])
print(D[0])
print(qqq)
E = u[1].flatten().reshape(n ** 2, 1)
E = np.matmul(E, E.T)
E2 = u2[1].flatten().reshape(n ** 2, 1)
E2 = np.matmul(E2, E2.T)

v,s,v1=np.linalg.svd(E)
v,s2,v1=np.linalg.svd(u2[1])
print(np.max(s2))
print(np.max(s))
print(q)


# +np.cos(c2*T)*np.sin(math.pi*kx2*X)*np.sin(math.pi*ky2*Y) \
# +np.cos(c3*T)*np.sin(math.pi*kx3*X)*np.sin(math.pi*ky3*Y)



resp=[]
resq=[]
resl=[]
for i in range(len(t)):
    E= u[i].flatten().reshape(n ** 2, 1)
    E=np.matmul(E,E.T)
    # E=u[i]

    p=polar(E, side='left')[1]
    q = polar(E, side='right')[0]
    if i==0:
        p0=p
        q0=q
        E0=E


    # resp.append(frob_norm(p,p*0))
    # resp.append(np.max(np.linalg.eigh(p)[0])/(l0))
    resq.append(abs(frob_norm(q,E)-frob_norm(q0,E)))
    # resp.append(frob_norm(p, 0)/144)




print(c)
# plt.plot(resp, 'r', label='p')
plt.plot(t,resq, 'g', label='q')
# plt.plot(resl, 'b', label='relation')
plt.legend()
plt.show()
    # resq.append(np.sqrt(frob_norm(q,u)))
    # resp.append(np.max(np.linalg.eigh(u)[0]))
    # resp.append(np.sqrt(frob_norm(p,p*0)))
    # resp.append(np.linalg.(p))
print(qq)
# print(np.cos(2*t))
# print(ii)
# print(np.sin(3*t))
# print(q)


# u0=(np.sin(8*x)+np.sin(9*x)).reshape(n-1,1)
#
# q0,p0=polar(u0,side='left')
#
# q0,p0=polar(np.multiply(u0,u0.T),side='left')

# print(np.multiply(p0,q0))


# u,s,v=np.linalg.svd(u0,full_matrices=True, compute_uv=True
#                     )
# # print(np.multiply(np.multiply(u,s),v.T))
# print(np.multiply(u0,u0.T))
# # print(np.multiply(p0, q0.T))
#
# print(aa)
resq=[]
resp=[]
# print(q0)

# for k in np.arange(0,len(t),1):
#     u=(np.cos(8*t[k])*np.sin(8*x)+np.cos(9*t[k])*np.sin(9*x)).reshape(n-1,1)
#
#
#     # u = np.multiply(A, u),
#     # q,p=polar(u,side='left')
#
#     q,p=polar(np.matmul(u,u.T),side='left')
#
#     # print(p0)
#     print(np.max(abs(np.matmul(q0.T,q)-np.eye(n-1))))



    # resq.append(np.sqrt(frob_norm(q,u)))
    # resp.append(np.max(np.linalg.eigh(u)[0]))
    # resp.append(np.sqrt(frob_norm(p,p*0)))
    # resp.append(np.linalg.(p))



plt.plot(resp,'r',label='p')
plt.plot(resq,'g',label='q')
plt.legend()
plt.show()
print(qw)



# A=np.array([-1,1,0],[0,-1,1],[0,0,1])
print(q)
from DRP_multiple_networks.constants import Constants


from DRP_multiple_networks.data_generator import create_test_data
from DRP_multiple_networks.drp import calculate_DRP
from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2


def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)


# a=np.array([[0.5,1,0],[0.5,1,1],[0,1,0.5]])
# u=np.array([[1,1,1],[2,2,2],[3,3,3]])
# print(frob_norm(polar(np.matmul(a, u))[0], u))
# print(frob_norm(polar(u)[0], u))

c=0.1

# m0=0.5*np.array([[0.5,1,4.5,0.5],[2,3,2,2],[3,3,5,0],[4,0,0,4]])

m0=0.5*np.array([[1]*4,[2]*4,[3]*4,[4]*4])
v0,s0,u0=np.linalg.svd(m0,full_matrices=True, compute_uv=True)


d=np.array([[-2,1.,0,0],[1,-2.,1,0],[0,1.,-2,1],[0,0,1.,-2]])
print(polar(d)[0])
print(q)
# print(np.linalg.det(u0))
# print(np.linalg.svd(a,full_matrices=True, compute_uv=True)[1])
a=np.eye(4) + 0.25 * d
print(polar(d)[0])
print(q)
mat=np.array([[2,0,0,0],[0,2.,0,0],[0,0.,2,0],[0,0,0,2]])
m1=np.matmul(mat,m0)
# m1=2*m0
# print(frob_norm(u1, polar(u1)[0]))
# print(frob_norm(u0, polar(u0)[0]))


v1,s1,u1=np.linalg.svd(m1,full_matrices=True, compute_uv=True)
# print(frob_norm(s0-1,0))
# print(s1)

print(frob_norm(m0, polar(m0, side='left')[0]))
print(frob_norm(m1, polar(m1, side='left')[0]))
# n=0.5
# x=np.linspace(math.pi/2,math.pi,1200)
# f=(np.sin(x*n+x)-np.sin(x*n-x))/(2*x)-np.cos(x)
# plt.plot(x,f)
# plt.show()

# path = Constants.PATH
#
# name = 'test_model1'
# saving_path = path + 'Experiment_' + name + '_details/'
# model1 = keras.models.load_model(saving_path + 'model.pkl',
#                                  custom_objects={'custom_loss': custom_loss, 'custom_loss3': custom_loss3})
# model1.load_weights(
#     saving_path + 'model_weights_val_number_' + str(0) + '.pkl').expect_partial()  # print(model1.trainable_weights)
#
# l_model = []
# l_fourth = []
# l_drp = []
#
# for N in [21,23]:
#     time_steps=(N-1)*2+1
#     h=1/(N-1)
#     t_f=0.01
#     dt=t_f/(time_steps-1)
#     cfl=dt/h
#     x1 = np.linspace(0., 1, N)
#     x2 = np.linspace(0., 1, N)
#     create_test_data(options='lt', h=h, dt=dt, t_f=t_f, time_steps=time_steps,
#                      X1=x1, X2=x2)
#     with open(path + 'test/test_data.pkl', 'rb') as file:
#         test_data = pickle.load(file)
#
#     data = {name: test_data[name][0] for name in list(test_data)}
#     print(time_steps)
#
#     for i in range(len(test_data['e'])):
#         data = {name: test_data[name][i] for name in list(test_data)}
#         loss_yee4('4order',0., -1 / 24, 0.,data, t_steps=time_steps, cfl=cfl)
#         # l_model.append(
#         #     loss_yee('model', model1.trainable_weights[0], model1.trainable_weights[1], model1.trainable_weights[2],
#         #              data))
# #
# print(q)
#
#
#
#
# for i in range(len(test_data['e'])):
#     data = {name: test_data[name][i] for name in list(test_data)}
#
#     l_fourth.append(loss_yee4('4order', 0., -1 / 24, 0., data))
#     l_model.append(
#         loss_yee4('model', model1.trainable_weights[0], model1.trainable_weights[1], model1.trainable_weights[2], data))
#     # l_yee.append(loss_yee('Yee', 0, 0, data))
#     # l_drp.append(loss_yee('DRP', 0., var, 0, data))
#     # print(np.log(l_drp))
#
#     print(l_model)
#     # print(l_drp)
#     print(l_fourth)
