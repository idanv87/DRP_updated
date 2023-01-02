import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from scipy.linalg import polar
from scipy.linalg import circulant
from DRP_multiple_networks.auxilary.aux_functions import relative_norm, frob_norm, max_norm

t=np.linspace(0,(2*np.sqrt(2)/5),73)

n=15
# x=np.linspace(0,math.pi,n+1)[:-1]
# h=math.pi/n
# f=np.array([1]+[0 for k in range(40)]+[-1])
# print(np.fft.fft(f))
# print(dd)

kernel=np.zeros((n,1))
kernel[0]=-1
kernel[-1]=1

A=circulant(kernel)[:-1]

u,s,vn=np.linalg.svd(A)
print(s)
print(dd)
err=[]
err2=[]
err3=[]

k1=np.arange(1,20,1)
k2=0

for k in k1:
    np.sin(k * x) / k * np.sqrt(np.sum(np.sin(k * x) ** 2))
    f=(np.sin(k*x)+np.sin(k2*x)).reshape(n,1)/100
    ftilde=np.matmul(A, f)/100
    ftagdiff=(k*np.cos(k*(x+h/2))+k2*np.cos(k2*(x+h/2))).reshape(n,1)/100
    f=np.matmul(f,f.T)
    ftilde = np.matmul(ftilde, ftilde.T)
    ftagdiff=np.matmul(ftagdiff,ftagdiff.T)
    norm=np.sqrt(frob_norm(ftagdiff,0))
    normtild=np.sqrt(frob_norm(ftilde,0))

    u,s,v=np.linalg.svd(ftagdiff, full_matrices=False, compute_uv=True )
    u, s2, v = np.linalg.svd(ftagdiff, full_matrices=False, compute_uv=True)
    # s[1:]=0
    fnew=np.matmul(u,np.matmul(np.diag(s), v))

    # err3.append(normtild)
    err.append(frob_norm(ftilde , ftagdiff))
    # err2.append(frob_norm(ftilde*norm/normtild, ftagdiff))
    err2.append(frob_norm(fnew, ftagdiff))







plt.plot(err,'r')
plt.plot(err2)
# plt.plot(err3)
plt.show()
print(q)

# k=np.arange(11,20,1)
# res=[]
# for k1 in [10]:
#     x = np.linspace(0, 1, 21)
#     dx = x[1] - x[0]
#     z = np.linspace(0, 1, 201)
#     plt.scatter(x, np.sin(k1 * math.pi * x),label=str(k1))
#     plt.plot(z, np.sin(k1 * math.pi * z),label=str(k1))
#     print(np.sin(k1 * math.pi * x))

    # y = (x + dx / 2)[0:-1]
    # f = np.sin(k1 * math.pi * x)
    # ftag = k1 * math.pi * np.cos(math.pi * k1 * y)
    # res.append(np.max(abs((np.diff(f) / dx-ftag)/(ftag))))
    # plt.plot(((((np.diff(f) / dx  - ftag)/(ftag))**2)))
# plt.legend()
plt.plot(res)
plt.show()







# print(np.linalg.matrix_rank(u))

# print(np.matmul(np.matmul(u,np.diag(s)),v)-f)

# A = tf.constant([[0, 0.001],[0.001,0.001]], shape=[2,2], dtype=tf.float64)
# B = tf.constant([[2, 1],[2,9]], shape=[2,2], dtype=tf.float64)
# u,d,v=np.linalg.svd(A)
# # print(np.sum(d**2))
# Q=polar(A)[0]
# P=polar(A)[1]
# P2=polar(np.matmul(B,A))[1]
# print(np.sum(Q**2))
# print(d)
#

print(qq)


# d, u, v = tf.linalg.svd(A, full_matrices=False, compute_uv=True)
# print(tf.matmul(u,tf.transpose(v))-Q)



# restf=tf.matmul(u,v)




# print(tf.matmul(tf.matmul(u,d),v))

# figure eror quotient :
# dr_calculator()
# print(q)


# n = [21, 41, 81, 161, 321]
# t = [2 / (18 * 2 ** 0.5), 2 / (36 * 2 ** 0.5), 2 / (72 * 2 ** 0.5), 2 / (144 * 2 ** 0.5), 2 / (288 * 2 ** 0.5)]
# x = [1, 1, 1, 1, 1]
# time_steps = [21, 21, 21, 21, 21]
# k_test = [18, 36, 72, 144, 288]
# solve_equation(n, x, t, time_steps, k_test)
# solve_equation(n, x, t, time_steps, k_test)
# # error_calculator(quotient=0, k_test=k_test)
# # error_calculator(quotient=1, k_test=k_test)
# print(q)
#################################


####################################
# figure eror quotient2 :
# var=-0.09053062
# n = [21, 41, 81, 161, 321]
# t = [2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5), 2 / (18 * 2 ** 0.5) ]
# x = [1, 1, 1, 1, 1]
# time_steps = [21, 41, 81, 161, 321]
# k_test = [18, 36, 72, 144, 288]
#
# solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# error_calculator(quotient=1, k_test=k_test)

#######################################
# figure eror figure 3 :
# var=-0.09053188
# n = [21, 81,81,81]*40
# t = [2 / (5 * 2 ** 0.5)]*40
# x = [1]*40
# time_steps = [21, 145,145,145]
# k_test = np.arange(20,40,1)
# # dr_calculator(n,x,t,time_steps,k_test)
# #
# solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# print(q)
#######################################
# print(q)
#
# k=23*2**0.5
# y=np.linspace(0,2/184/2**0.5,210)
# x=np.linspace(0,2/184/2**0.5,81)
# plt.scatter(x,np.cos((math.pi*x*k)))
# plt.plot(y,np.cos((math.pi*y*k)))
# plt.show()
# n = [161]*80
# t = [1 / (20 * 2 ** 0.5)] * 80
# time_steps = n
# k_test = np.arange(80,160,1)
# dr_calculator(n,t,time_steps,k_test)


#
# from drp import calculate_DRP
# from DRP_multiple_networks.auxilary.drp2 import calculate_DRP2
# from DRP_multiple_networks.auxilary.aux_functions import relative_norm

# #
# import matplotlib.pyplot as plt
# # from DRP_multiple_networks import constants
#
#
# # x=calculate_DRP()
# # y=calculate_DRP2()
# # print(x)
# # print(y)
# # # print(Constants.CFL)
# # # # x=tf.constant([[1,2],[1,2]])
# # # # print(relative_norm(x,x ))
# # # # print(tf.math.pow(x,2))
# # X=np.arange(10, 19, 1)**2
# # Y=np.arange(1,20,1)**2
# # Z=Y
# # for x in X:
# #     for y in X:
# #         print(np.sqrt(x+y))
# #
# #
# # print(q)

# # # x=np.linspace(0,1,82)
# # # plt.plot(x,np.cos(math.pi*x*20))
# plt.show()
# # #
# # # print(Constants.CFL)
