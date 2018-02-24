import matplotlib.pyplot as plt
import numpy as np
import sys

from Regression.ronbun.ronbun_jikken3.make_communication import Circle_communication
from Regression.ronbun.ronbun_jikken3.Solver import Solver
from Regression.ronbun.ronbun_jikken3.algorithm import Distributed_solver
from Regression.ronbun.ronbun_jikken3.condition_check import *
np.random.seed(0)

test = True
#共通parameter============================================================
patterns = 2
iteration = 100000
n = 4
m = 2
K = 1 #先行研究でのellに相当(量子化レベル)
#####################################################################################
# f_i = ||Ax-b||_2^2の場合
A1 =  np.array([[0.7,0.4],[0.3,0.6]])

alpha = np.linalg.norm(np.dot(A1.T,A1))
beta = np.linalg.norm(np.dot(A1.T,A1))
print(alpha,beta)
C_g = 1.5
#####################################################################################

# ==========================================================================
#提案手法parameter===========================================================
mu= 0.999935
C_x=0.5
C_v=0.45

w = 0.0005
eta = 0.0002

delta_ast = (1.5 ** 2 + 1.5 ** 2) ** 0.5 * n ** 0.5
delta_v = (1 ** 2 + 1 ** 2) ** 0.5
delta_x = 0

C_x0 = 0 #max_i ||x_i(0)||
C_v0 = 1.5 #max_i ||v_i(0)||
# =========================================================================

#先行研究(劣勾配)============================================================
w_2 = 0.014
s_0 = 10
gamma = 0.99

C_x2 = 0 #max_i ||x_i(0)||
C_delta2 = 0 #max_i,j ||x_i(0)-x_j(0)||
# ============================================================================

Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Graph_2 = Circle_communication(n,w_2)
Graph_2.make_circle_graph()
Laplacian_matrix = Graph_2.send_L()

Condition_proposed(n, m, Weight_matrix, w, eta, C_x,C_v, mu, alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0)
Condition_prior(n,m,Laplacian_matrix,w_2,s_0,gamma,C_x2,C_delta2,C_g,K)

w_2 = 0.014
gamma = 0.99
C_g = 1.5
Graph_2 = Circle_communication(n,w_2)
Graph_2.make_circle_graph()
# Weight_matrix_2 = Graph_2.send_P()
Laplacian_matrix = Graph_2.send_L()

if test is False:
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


Method = ['Proposed','Subgrad']
# Method = ['Proposed','Subgrad']
Result_data = {}
# Method = ['Subgrad']
for algo in Method:
    if algo == 'Proposed':
        other_param = (eta,mu,C_x,C_v)
        weight_matrix = Weight_matrix
    elif algo == 'Subgrad':
        other_param = None
        weight_matrix = Weight_matrix_2

    D_sol = Distributed_solver(n,m,A,b,weight_matrix,algo,iteration,other_param)
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
