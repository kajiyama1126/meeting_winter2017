import sys

import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar

from Regression.shuron.shuron_jikken2.make_communication import Circle_communication
from Regression.shuron.shuron_jikken2.shuron_jikken2_agent import Agent_YiHong14
from Regression.shuron.shuron_jikken2.shuron_jikken2_agent import \
    Agent_harnessing_quantize_add_send_data_shuron_jikken2 as Agent_jikken2

np.random.seed(0)

# test = False
# test_sub = False
test = True
test_sub = True

n = 4
m = 2

mu_x = 0.999935
h_0 = 0.5
C_h = 0.45
w = 0.0005
eta = 0.0002

iteration = 200000
patterns = 2

alpha = 1
beta = 1

delta_ast = (1.5 ** 2 + 1.5 ** 2) ** 0.5 * n ** 0.5
delta_v = (1 ** 2 + 1 ** 2) ** 0.5
delta_x = 0

C_x0 = 0
C_v0 = 1.5
d_hat = 2
Graph = Circle_communication(n, w)
Graph.make_circle_graph()
weight_matrix = Graph.send_P()
print(weight_matrix)

def Condition_proposed(n, m, weight_matrix, w, eta, h_0, mu_x, C_h, alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0):
    d_hat = np.linalg.norm(np.sign(weight_matrix), 1) - 1

    Peron_matrix = weight_matrix - (1 / n) * np.ones([n, n])
    sigma = np.linalg.norm(Peron_matrix, 2)
    G = np.array([[sigma + beta * eta, beta * (beta * eta + 2 * d_hat * w), eta * beta * beta], [eta, sigma, 0],
                  [0, eta * beta, 1 - alpha * eta]])

    l, p = np.linalg.eig(G)
    rho = max(l)
    P = np.transpose(p)
    P_inv = np.linalg.inv(P)
    C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)

    if l[1] == l[2]:
        C_theta = 3 / (rho - l[1])
    else:
        C_theta = 1

    C_1 = C_P * C_theta * ((delta_v ** 2 + delta_x ** 2 + delta_ast ** 2)) ** 0.5
    Gamma = (C_P * C_theta * d_hat * w * (m * n * ((beta + 1. / C_h) ** 2 + 1)) ** 0.5) / (mu_x - rho)
    C_x = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 + 2 * eta * beta * ((1 / n) ** 0.5) + eta)
    C_v = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 * beta * (m ** 0.5) + (2 * eta * beta ** 2) * (m / n) ** 0.5 + w * (
            2 * d_hat * (d_hat + 1)) ** 0.5 + eta * beta * (m ** 0.5))
    ell_x = C_x * (C_1 / (mu_x * h_0) + Gamma / (mu_x)) + (2 * w * d_hat + 1) / (2 * mu_x) - 1 / 2
    ell_v = C_v * (C_1 * C_h / (mu_x * h_0) + C_h * Gamma / (mu_x)) + (m ** 0.5 * d_hat * beta * w * C_h) / mu_x + (
            2 * d_hat * w + 1) / (2 * mu_x) - 1 / 2

    print('ell_x0', C_x0 / h_0 - 1 / 2, 'ell_v0', C_v0 / (h_0 * 1 / C_h) - 1 / 2)
    print('ell_x', ell_x, 'ell_v', ell_v)

Condition_proposed(n, m, weight_matrix, w, eta, h_0, mu_x, C_h, alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0)
# rho = max(1 - alpha * eta / 2, sigma + 5 * ((eta * beta) ** 0.5) * ((beta / alpha) ** 0.5))
#
# G = np.array([[sigma + beta * eta, beta * (beta * eta + 2 * d_hat * w), eta * beta * beta], [eta, sigma, 0],
#               [0, eta * beta, 1 - alpha * eta]])
#
# # G_sym = Matrix(G)
# # P, J = G_sym.jordan_form()
# # print(P)
# # print(J)
# l, p = np.linalg.eig(G)
# P = np.transpose(p)
# P_inv = np.linalg.inv(P)
# C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)
# print('C_P', C_P)
# G_eig, G_eigen_vec = np.linalg.eig(G)
# rho = max(G_eig)
# print('mu_x', mu_x)
# print('rho', rho)
# print('mu_x-rho', mu_x - rho)
# print('eigen_vec', p)
#
# if G_eig[1] == G_eig[2]:
#     C_theta = 3 / (rho - G_eig[1])
# else:
#     C_theta = 1
# print('C_theta', C_theta)
#
# print(d_hat * w, 2 * eta * beta, eta)
# # upper_w = np.max(Weight_matrix)
# w = w
# C_1 = C_P * C_theta * ((delta_v ** 2 + delta_x ** 2 + delta_ast ** 2)) ** 0.5
# Gamma = (C_P * C_theta * d_hat * w * (m * n * ((beta + 1. / C_h) ** 2 + 1)) ** 0.5) / (mu_x - rho)
# C_x = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 + 2 * eta * beta * ((1 / n) ** 0.5) + eta)
# C_v = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 * beta * (m ** 0.5) + (2 * eta * beta ** 2) * (m / n) ** 0.5 + w * (
#             2 * d_hat * (d_hat + 1)) ** 0.5 + eta * beta * (m ** 0.5))
# ell_x = C_x * (C_1 / (mu_x * h_0) + Gamma / (mu_x)) + (2 * w * d_hat + 1) / (2 * mu_x) - 1 / 2
# ell_v = C_v * (C_1 * C_h / (mu_x * h_0) + C_h * Gamma / (mu_x)) + (m ** 0.5 * d_hat * beta * w * C_h) / mu_x + (
#             2 * d_hat * w + 1) / (2 * mu_x) - 1 / 2
#
# print('ell_x0', C_x0 / h_0 - 1 / 2, 'ell_v0', C_v0 / (h_0 * 1 / C_h) - 1 / 2)
# print('ell_x', ell_x, 'ell_v', ell_v)

if test is False:
    sys.exit()


w_2 = 0.014
gamma = 0.99
C_g = 1.5
Graph_2 = Circle_communication(n, w_2)
Graph_2.make_circle_graph()
Weight_matrix_2 = Graph_2.send_P()
Laplacian_matrix = Graph_2.send_L()
print(Weight_matrix_2, Laplacian_matrix)
l_2, p_2 = np.linalg.eig(Laplacian_matrix)

l_2.sort()
print(l_2, p_2)

# Peron_matrix_2 = Weight_matrix_2- (1/n)*np.ones([n,n])
# lamb_2 = np.linalg.norm(Peron_matrix_2, 2)

lamb_2 = l_2[1]
lamb_n = l_2[n - 1]
rho_h = 1 - w_2 * lamb_2
if gamma - rho_h < 0:
    print('error')
print(gamma - rho_h)
M1 = (1 + w_2 * 2 * d_hat) / (2 * gamma) + w_2 * C_g
M2 = ((m * n) ** 0.5 * w_2 ** 2 * lamb_n * (lamb_n + 2 * gamma * C_g)) / (2 * gamma * (gamma - rho_h))
M = M1 + M2
s0 = 0
print('M', M)

if test_sub is False:
    sys.exit()

A = None
B = []

for i in range(n):
    A = np.identity(m)
    b = np.array([-1.5, 0.5]) + 1.0 * np.random.rand(2)
    # if i == 0 or i ==1:
    #     b = np.array([-1,1,1])
    # elif i == 2:
    #     b =  np.array([1,0,-1])
    # elif i ==3:
    #     b = np.array([3, 1, -3])
    B.append(b)

ave_b = 0
for i in range(n):
    ave_b += B[i] / n

f_opt = 0
for i in range(n):
    f_opt += 1 / 2 * np.linalg.norm(ave_b - B[i], 2) ** 2
print(ave_b)

ydata_set = []
zdata_set = []

for pattern in range(patterns):
    Agents = []
    sumf_list = []
    prog = ProgressBar(max_value=iteration)
    for i in range(n):
        if pattern < int(patterns) / 2:
            # Agents.append(Agent_harnessing(n, m, A, B[i], eta, i, Weight_matrix[i]))
            Agents.append(Agent_jikken2(n, m, A, B[i], eta, weight_matrix[i],i, mu_x, C_h, h_0))
        else:
            Agents.append(Agent_YiHong14(n, m, A, B[i], Weight_matrix_2[i], i))

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
                f_sum += 1 / 2 * np.linalg.norm(Agents[i].x_i - B[j], 2) ** 2
            f_value.append(f_sum)

        sumf_list.append(max(f_value) - f_opt)
        # print(max(f_value)-f_opt)
        if k % 10000 == 0:
            print('iteration', k)
            for i in range(n):
                print(Agents[i].x_i)

    for i in range(n):
        print(Agents[i].x_i)

    dim_label = ['Proposed', '[26]']
    plt.plot(sumf_list, label=dim_label[pattern])
    plt.yscale('log')

    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='gray', linestyle=':',axis='y')
    plt.minorticks_on()
    plt.xlabel('iteration $k$', fontsize=14)
    plt.ylabel('$max_{i} f(x_i(k))-f^*$', fontsize=14)
    y_data, z_data = Agents[0].send_y_data_zdata()
    ydata_set.append(y_data)
    zdata_set.append(z_data)

plt.legend()
plt.show()

for pattern in range(patterns):
    y_data = ydata_set[pattern]
    z_data = zdata_set[pattern]

    for i in range(n):
        if y_data[i][0] == []:
            pass
        else:
            dim_label = ['1', '2']
            for j in range(m):
                plt.plot(y_data[i][j], 'o', markersize=4)
                plt.xlabel('iteration $k$', fontsize=18)
                plt.ylabel('$y_{12}^' + dim_label[j] + '$', fontsize=20)
                plt.legend()
                plt.show()
                # plt.legend()
                # plt.show()
                if z_data is not None:
                    plt.plot(z_data[i][j], 'o', markersize=4)
                    plt.xlabel('iteration $k$', fontsize=18)
                    plt.ylabel('$z_{12}^' + dim_label[j] + '$', fontsize=20)
                    plt.legend()
                    plt.show()
            break
    # plt.legend()
    # plt.show()
    # plt.legend()
    # plt.show()
