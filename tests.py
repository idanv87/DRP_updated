# figure eror quotient :
# dr_calculator()
# print(q)
n = [21, 41, 81, 161, 321]
t = [2 / (18 * 2 ** 0.5), 2 / (36 * 2 ** 0.5), 2 / (72 * 2 ** 0.5), 2 / (144 * 2 ** 0.5), 2 / (288 * 2 ** 0.5)]
x = [1, 1, 1, 1, 1]
time_steps = [21, 21, 21, 21, 21]
k_test = [18, 36, 72, 144, 288]
solve_equation(n, x, t, time_steps, k_test)
solve_equation(n, x, t, time_steps, k_test)
# error_calculator(quotient=0, k_test=k_test)
# error_calculator(quotient=1, k_test=k_test)
print(q)
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
