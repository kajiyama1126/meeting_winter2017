import matplotlib.pyplot as plt
import cvxpy as cvx
from progressbar import ProgressBar
import numpy as np

from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken4.condition_check import *
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken4.make_communication import Circle_communication
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken4.ronbun_jikken3_agent import Agent_harnessing_quantize_add_send_data_ronbun_jikken3 as Agent_jikken3

np.random.seed(0)
#Parameter_setting = True
Parameter_setting = False


def Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt):
    #iteration = 1000000
    iteration = 120000
    Agents = []
    sumf_list = []
    prog = ProgressBar(max_value=iteration)
    for i in range(n):
        Agents.append(Agent_jikken3(n, m, A[i], B[i], eta, Weight_matrix[i], i, C_x, C_v, mu))
    for k in range(iteration):
        prog.update(k + 1)

        for i in range(n):
            for j in range(n):
                x_j, name = Agents[i].send(j)
                Agents[j].receive(x_j, name)

        for i in range(n):
            Agents[i].update(k)

        f_value = []

        for i in range(n):
            f_sum = 0
            for j in range(n):
                f_sum += 1 / 2 * np.linalg.norm(np.dot(A[j], Agents[i].x_i) - B[j], 2) ** 2
            f_value.append(f_sum)

        sumf_list.append(max(f_value) - f_opt)
        # print(max(f_value)-f_opt)
        if max(f_value) - f_opt < 10 ** (-8):
            return sumf_list

        if k % 10000 == 0:
            print('iteration', k)
            for i in range(n):
                print(Agents[i].x_i)

def solve(A,B,m):

    x = cvx.Variable(m)
    obj = cvx.Minimize(0)
    for i in range(len(A)):
        obj += cvx.Minimize(1/2*cvx.power(cvx.norm(A[i]*x-B[i],2),2))
    pro = cvx.Problem(obj)
    pro.solve(abstol=1.0e-10, feastol=1.0e-10)
    return x.value, pro.value



# 共通parameter============================================================
patterns = 1
iteration = 400000
n = 4
m = 2
K = 1  # 先行研究でのellに相当(量子化レベル)
#####################################################################################
# f_i = ||Ax-b||_2^2の場合
A1 = np.array([[0.7, 0.2], [0.3, 0.6]])
# A1 = np.array([[1, 2], [1, 1]])
#b = np.array([-1.5, 0.5]) + 1.0 * np.random.rand(2)

alpha = np.linalg.norm(np.dot(A1.T, A1),2)
beta = np.linalg.norm(np.dot(A1.T, A1),2)
print(alpha, beta)
C_g = np.linalg.norm(np.dot(np.dot(A1.T, A1),np.array([-1,1])),2)
#####################################################################################

A = []
B = []
Result = []
for i in range(n):
    A.append(A1)
    b = np.array([-1, 0]) + 1.0 * np.random.rand(2)
    # if i == 0 or i ==1:
    #     b = np.array([-1,1,1])
    # elif i == 2:
    #     b =  np.array([1,0,-1])
    # elif i ==3:
    #     b = np.array([3, 1, -3])
    B.append(b)

x_opt,f_opt = solve(A,B,m)

# ==========================================================================
#5段階
# 提案手法parameter===========================================================
mu = 0.9999
C_x = 0.45
C_v = 0.90

w = 0.001
eta = 0.000555

# mu = 0.99992
# C_x = 0.45
# C_v = 0.9
#
# w = 0.0008
#
# eta = 0.00038

# mu = 0.99992
# C_x = 0.45
# C_v = 0.9
#
# w = 0.0008
# eta = 0.00038

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# =========================================================================

Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)
Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))
#7段階
# 提案手法parameter===========================================================
mu = 0.9997
C_x = 0.77
C_v = 1.56

w = 0.002
eta = 0.00088
#eta = 0.0011

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# =========================================================================

Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)
Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))

#15段階
# 提案手法parameter===========================================================
mu = 0.9994
C_x = 0.77
C_v = 1.56

w = 0.0063
eta = 0.0028

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# =========================================================================
Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)

Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))

#31段階
# 提案手法parameter===========================================================
mu = 0.9985
C_x = 0.82
C_v = 1.61

w = 0.011
eta = 0.0051

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# =========================================================================


Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()



Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)

Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))
#63段階
# 提案手法parameter===========================================================
mu = 0.9968
C_x = 2.1
C_v = 4.1

w = 0.021
eta = 0.0088

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# =========================================================================

Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()



Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)

Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))

# #1023段階
# #1023段階
# # 提案手法parameter===========================================================
# mu = 0.9935
# C_x = 0.0128
# C_v = 0.0251
#
# w = 0.043
# eta = 0.012
# delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
# delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
# delta_x = 0
#
# C_x0 = 0  # max_i ||x_i(0)||
# C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# # =========================================================================
#
# Graph = Circle_communication(n, w)
# Graph.make_circle_graph()
# Weight_matrix = Graph.send_P()


#
# Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)
#
# Result.append(Trail(n, m, A, B, eta, Weight_matrix, C_x, C_v, mu, f_opt))

dim_label = ['5-Level','7-Level','15-Level','31-Level','63-Level']
for i in range(len(dim_label)):
    plt.plot(Result[i], label=dim_label[i])
plt.yscale('log')
plt.xlabel('iteration $k$', fontsize=12)
plt.ylabel('$max_{i} f(x_i(k))-f^*$', fontsize=12)
#plt.tick_params(labelsize=10)
#plt.legend(fontsize=10)
#plt.tight_layout()
plt.grid(which='major', color='gray', linestyle=':')
plt.grid(which='minor', color='gray', linestyle=':', axis='y')
plt.minorticks_on()
plt.xlim([0, 120000])
# plt.xlim([0, 100000])
plt.ylim([0.00000001, 10])
plt.legend()
plt.savefig("cost_quantization_level.png")
plt.savefig("cost_quantization_level.eps")
plt.show()

