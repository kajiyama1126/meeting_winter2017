import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix
import sys
from progressbar import ProgressBar
from Regression.shuron_jikken2.shuron_jikken2_agent import Agent_harnessing_quantize_add_send_data_shuron_jikken2 as Agent_jikken2
from Regression.shuron_jikken2.shuron_jikken2_agent import Agent_YiHong14
from Regression.shuron_jikken2.make_communication import Circle_communication
from agent.agent import Agent_harnessing
test = False
# test = True
n = 4
m = 2
iteration = 2000
patterns = 1

alpha = 1
beta = 1

delta_ast = (1.5**2 + 1.5**2)**0.5 *n**0.5
delta_v = (1**2+1**2)**0.5
delta_x = 0

mu_x = 0.99993
h_0 = 10
C_h = 0.45
w = 0.0005
d_hat = 2
Graph = Circle_communication(n,w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()
print(Weight_matrix)

Peron_matrix = Weight_matrix- (1/n)*np.ones([n,n])

sigma = np.linalg.norm(Peron_matrix, 2)

# print('sigma',sigma)
print('sigma',sigma)

eta = 0.0002


n_w =(n-1)/n
# rho = max(1 - alpha * eta / 2, sigma + 5 * ((eta * beta) ** 0.5) * ((beta / alpha) ** 0.5))

G = np.array([[sigma + beta * eta, beta * (beta * eta + 2*d_hat*w), eta * beta * beta], [eta, sigma, 0],
              [0, eta * beta, 1 - alpha * eta]])

# G_sym = Matrix(G)
# P, J = G_sym.jordan_form()
# print(P)
# print(J)
l,p = np.linalg.eig(G)
P = np.transpose(p)
P_inv = np.linalg.inv(P)
C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)
print('C_P',C_P)
G_eig,G_eigen_vec = np.linalg.eig(G)
rho = max(G_eig)
print('mu_x',mu_x)
print('rho',rho)
print('mu_x-rho',mu_x-rho)
print('eigen_vec',p)

if G_eig[1] == G_eig[2]:
    C_theta = 3/(rho-G_eig[1])
else:
    C_theta = 1
print('C_theta',C_theta)

print(d_hat * w,2*eta*beta,eta)
# upper_w = np.max(Weight_matrix)
w=w
C_1 =C_P * C_theta * ((delta_v**2 + delta_x**2 + delta_ast**2))**0.5
Gamma_x = (C_P * C_theta * d_hat * w * (m * n * ((beta** 2+1))) ** 0.5) / (mu_x - rho)
Gamma_v = (C_P * C_theta * d_hat * w * (m * n) ** 0.5) / (mu_x - rho)
C_x = (2*d_hat * w*n_w +  2*eta * beta * ((1/n) ** 0.5 )+ eta)
C_v = (2*d_hat * w*n_w * beta * m ** 0.5 + (2 * eta * beta ** 2)*(m/n)**0.5 + 2*d_hat * w *n_w+ eta * beta * m ** 0.5)
ell_x = C_x * (C_1/(mu_x * h_0) + Gamma_x/(mu_x) + Gamma_v/(C_h*mu_x)) + (2 * d_hat * w*n_w + 1) / (2 * mu_x) - 1 / 2
ell_v= C_v * (C_1*C_h/(mu_x * h_0) + C_h*Gamma_x/(mu_x) + Gamma_v/(mu_x)) + (m ** 0.5 * d_hat * beta * w * C_h) / mu_x + (2 * d_hat * w + 1) / (2 * mu_x) - 1 / 2


w_2 = 0.004
Graph_2 = Circle_communication(n,w_2)
Graph_2.make_circle_graph()
Weight_matrix_2 = Graph.send_P()
print(Weight_matrix_2)

Peron_matrix_2 = Weight_matrix_2- (1/n)*np.ones([n,n])
lamb_2 = np.linalg.norm(Peron_matrix_2, 2)




if test is True:
    sys.exit()


A = None
B = []
Agents = []
sumf_list = []
for i in range(n):
    A = np.identity(m)
    b= np.array([-1.5,0.5]) + 1.0*np.random.rand(2)
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
    f_opt += 1/2*np.linalg.norm(ave_b-B[i],2)**2
print(ave_b)
for pattern in range(patterns):
    prog = ProgressBar(max_value=iteration)
    for i in range(n):
        if pattern < int(patterns):
            # Agents.append(Agent_harnessing(n, m, A, B[i], eta, i, Weight_matrix[i]))
            Agents.append(Agent_jikken2(n, m, A, B[i], eta, i, Weight_matrix[i],mu_x,C_h,h_0))
        # else:
        #     # Agents.append(Agent_YiHong14(n, m, A, b, eta, i, Weight_matrix[i]))


    for k in range(iteration):
        prog.update(k+1)

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
                f_sum +=1/2* np.linalg.norm(Agents[i].x_i - B[j],2)**2
            f_value.append(f_sum)

        sumf_list.append(max(f_value)-f_opt)
        # print(max(f_value)-f_opt)
        if k % 10000 == 0:
            print('iteration',k)
            for i in range(n):
                print(Agents[i].x_i)


    for i in range(n):
        print(Agents[i].x_i)

    plt.plot(sumf_list)
    plt.yscale('log')
    plt.show()

    y_data, z_data = Agents[0].send_y_data_zdata()

    for i in range(n):
        if y_data[i][0] == []:
            pass
        else:
            dim_label = ['1','2']
            for j in range(m):
                plt.plot(y_data[i][j], 'o' ,label='$||y_{ij}||_\infty$' + dim_label[j],markersize=4)
                # plt.legend()
                # plt.show()
                plt.plot(z_data[i][j], 'o',label='$||z_{ij}||_\infty$'+ dim_label[j],markersize=4)
            plt.legend()
            plt.show()
        break
    # plt.legend()
    # plt.show()
