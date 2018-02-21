import matplotlib.pyplot as plt
import numpy as np
import sys

from Regression.ronbun.ronbun_jikken3.make_communication import Circle_communication
from Regression.ronbun.ronbun_jikken3.Solver import Solver
from Regression.ronbun.ronbun_jikken3.algorithm import Distributed_solver
from Regression.ronbun.ronbun_jikken3.condition_check import *
np.random.seed(0)

# test = False
# test_sub = False
test = True
test_sub = True

n = 4
m = 2

mu= 0.999935
C_x=0.5
C_v=0.45

w = 0.0005
eta = 0.0002

iteration = 200000
patterns = 2

delta_ast = (1.5 ** 2 + 1.5 ** 2) ** 0.5 * n ** 0.5
delta_v = (1 ** 2 + 1 ** 2) ** 0.5
delta_x = 0

C_x0 = 0
C_v0 = 1.5
d_hat = 2
Graph = Circle_communication(n, w)
Graph.make_circle_graph()
weight_matrix = Graph.send_P()

w_2 = 0.014
gamma = 0.99
Graph_2 = Circle_communication(n,w_2)
Graph_2.make_circle_graph()
Weight_matrix_2 = Graph_2.send_P()
Laplacian_matrix = Graph_2.send_L()

#####################################################################################
alpha = 1
beta = 1
C_g = 1.5
#####################################################################################


Condition_proposed(n, m, weight_matrix, w, eta, C_x,C_v, mu, alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0)
Condition_prior(n,m,Laplacian_matrix,w_2,gamma,C_g,d_hat)

w_2 = 0.014
gamma = 0.99
C_g = 1.5
Graph_2 = Circle_communication(n,w_2)
Graph_2.make_circle_graph()
Weight_matrix_2 = Graph_2.send_P()
Laplacian_matrix = Graph_2.send_L()



if test_sub is False:
    sys.exit()


A = []
b = []


for i in range(n):
    A_i =  np.array([[1,0.5],[0.3,1]])
    b_i= np.array([-1.5,0.5]) + 1.0*np.random.rand(2)
    A.append(A_i)
    b.append(b_i)

Sol = Solver(n,m,A,b)
f_opt,x_opt = Sol.send_opt()
rho = 1
resol = 3

Method = ['Proposed','Subgrad','ADMM']
# Method = ['Proposed','Subgrad']
Result_data = {}
# Method = ['Subgrad']
for algo in Method:
    if algo == 'Proposed':
        other_param = (eta,mu_x,C_h,h_0)
    elif algo == 'Subgrad':
        other_param = None
    elif algo == 'ADMM':
        other_param = (rho,resol)

    D_sol = Distributed_solver(n,m,A,b,Weight_matrix,algo,iteration,other_param)
    D_sol.Make_agent()
    result = D_sol.Iteration(f_opt)
    Result_data[algo] = result
for algo in Method:
    plt.plot(Result_data[algo],label = algo)
plt.yscale('log')
plt.legend()
plt.show()



# ydata_set = []
# zdata_set = []
#
#
#     dim_label = ['Proposed algorithm', 'Prior algorithm']
#     plt.plot(sumf_list,label=dim_label[pattern])
#     plt.yscale('log')
#     plt.xlabel('iteration $k$',fontsize=14)
#     plt.ylabel('$max_{i} f(x_i(k))-f^*$', fontsize=14)
#     y_data, z_data = Agents[0].send_y_data_zdata()
#     ydata_set.append(y_data)
#     zdata_set.append(z_data)
#
# plt.legend()
# plt.show()
#
#
# for pattern in range(patterns):
#     y_data=ydata_set[pattern]
#     z_data=zdata_set[pattern]
#
#     for i in range(n):
#         if y_data[i][0] == []:
#             pass
#         else:
#             dim_label = ['1','2']
#             for j in range(m):
#                 plt.plot(y_data[i][j], 'o' ,markersize=4)
#                 plt.xlabel('iteration $k$',fontsize=16)
#                 plt.ylabel('$y_{ij}^'+dim_label[j]+'$',fontsize=18)
#                 plt.legend()
#                 plt.show()
#                 # plt.legend()
#                 # plt.show()
#                 if z_data is not None:
#                     plt.plot(z_data[i][j], 'o',markersize=4)
#                     plt.xlabel('iteration $k$',fontsize=16)
#                     plt.ylabel('$z_{ij}^'+dim_label[j]+'$',fontsize=18)
#                     plt.legend()
#                     plt.show()
#             break
#     # plt.legend()
#     # plt.show()
#     # plt.legend()
#     # plt.show()
