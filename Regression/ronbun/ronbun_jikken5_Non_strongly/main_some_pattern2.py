import numpy as np
from Regression.ronbun.ronbun_jikken5_Non_strongly.non_convex_agent import Agent_harnessing_nonconvex_quantize_add_send_data,Agent_harnessing_nonconvex,Agent_harnessing_nonconvex_quantize_add_send_data_alpha,Agent_harnessing_nonconvex_quantize_add_send_data_alpha2
from Regression.ronbun.ronbun_jikken5_Non_strongly.make_communication import Communication
import matplotlib.pyplot as plt
from Regression.ronbun.ronbun_jikken5_Non_strongly.Solver import Solver,Log_Solver
import math


np.random.seed(1)

n =20
m =4
iteration = 5000
# eta = 0.005
#
#alpha_pattern = [0.01,0.005,0.003,0.002,0.001]
alpha_pattern = [0.002, 0.001, 0.0005, 0.0001]
# alpha_pattern = [0.01]
eta_pattern = alpha_pattern
a_i_pattern = [1,1,1,1,1,1,1,1,1,1,
               -1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

# alpha = 0.005
Graph = Communication(n, 4, 0.3)
Graph.make_connected_WS_graph()
Weight_matrix = Graph.send_P()

# A =None

# A = np.array([2,3,-1,3])
A= []
B = []
# Agents = []
sumf_list = [[] for i in range(len(alpha_pattern))]
for i in range(n):
    a = np.random.rand(m)
    b = np.random.rand(1)
    A.append(a)
    B.append(b)
    # Agents.append(Agent_harnessing_nonconvex(n, m,A ,b , eta, i, Weight_matrix[i]))
    # Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data_alpha(n, m,A ,b , eta,  Weight_matrix[i],i,alpha))
    # Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data(n, m, A, b, i, eta))
# sol_x = Solver(n,m,A,B)
sol_x = Log_Solver(n,m,A,B,a_i_pattern)
f_opt,x_opt = sol_x.send_opt()
# print(Agents[0].eta)

# alpha_pattern = [0.005,0.004,0.003,0.002,0.001]
# alpha_pattern = [0.01]
# eta_pattern = alpha_pattern

for pattern in range(len(alpha_pattern)):
    Agents = []
    alpha = alpha_pattern[pattern]
    eta = 0.1

    for i in range(n):
        a_i = a_i_pattern[i]
        a = A[i]
        b = B[i]
        Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data_alpha2(n, m,a,b , eta,  Weight_matrix[i],i,alpha,a_i))

    for k in range(iteration):
        for i in range(n):
            for j in range(n):
                x_j,name = Agents[i].send(j)
                Agents[j].receive(x_j, name)

        for i in range(n):
            Agents[i].update(k)

        sumf = 0

        x = Agents[0].x_i
        for i in range(n):
            a_i = a_i_pattern[i]
            sumf +=  math.log(1.+math.e**(a_i*(np.dot(A[i],x)-B[i])))
            # for j in range(m):
            #     if abs(x[j]-B[i][j])<1:
            #         sumf+= 1/4*(x[j]-B[i][j])**4
            #     else:
            #         sumf+=abs(x[j]-B[i][j])-3/4
        sumf_list[pattern].append(sumf-f_opt)


    # for i in range(n):
    #     x0 = np.linspace(0, 10)
    #     x1 = (Agents[i].x_i[0] * x0 + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
    #     im = plt.plot(x0, x1)
    #     ims[i].append(im)
    # print(k)
# for i in range(n):
#     print(Agents[i].x_i)

print(sum(B)/n)
# xaxis = np.linspace(0,iteration)
for pattern in range(len(alpha_pattern)):
    plt.plot(sumf_list[pattern],label=r'$\tilde{\alpha}$ =' + str(alpha_pattern[pattern]) )
plt.legend()
plt.yscale('log')
plt.grid(which='major', color='gray', linestyle=':')
plt.grid(which='minor', color='gray', linestyle=':')
plt.xlabel('iteration $k$', fontsize=10)
plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=10)
plt.xlim([0, iteration])
#plt.ylim([0.00001, 100])
plt.savefig("cost_compare_nonstrongly2.png")
plt.savefig("cost_compare_nonstrongly2.eps")
plt.show()