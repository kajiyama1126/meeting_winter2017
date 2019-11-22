import numpy as np
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken5_Non_strongly.non_convex_agent import Agent_harnessing_nonconvex_quantize_add_send_data,Agent_harnessing_nonconvex,Agent_harnessing_nonconvex_quantize_add_send_data_alpha
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken5_Non_strongly.make_communication import Communication
import matplotlib.pyplot as plt
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken5_Non_strongly.Solver import Solver


n =20
m =3
iteration = 5000
eta = 0.005

alpha = 0.005
Graph = Communication(n, 4, 0.3)
Graph.make_connected_WS_graph()
Weight_matrix = Graph.send_P()

A =None
B = []
Agents = []
sumf_list = []
for i in range(n):
    b = np.random.rand(m)
    B.append(b)
    # Agents.append(Agent_harnessing_nonconvex(n, m,A ,b , eta, i, Weight_matrix[i]))
    Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data_alpha(n, m,A ,b , eta,  Weight_matrix[i],i,alpha))
    # Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data(n, m, A, b, i, eta))

sol_x = Solver(n,m,A,B)
f_opt,x_opt = sol_x.send_opt()
print(Agents[0].eta)
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
        for j in range(m):
            if abs(x[j]-B[i][j])<1:
                sumf+= 1/4*(x[j]-B[i][j])**4
            else:
                sumf+=abs(x[j]-B[i][j])-3/4
    sumf_list.append(sumf-f_opt)


    # for i in range(n):
    #     x0 = np.linspace(0, 10)
    #     x1 = (Agents[i].x_i[0] * x0 + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
    #     im = plt.plot(x0, x1)
    #     ims[i].append(im)
    # print(k)
for i in range(n):
    print(Agents[i].x_i)

print(sum(B)/n)
# xaxis = np.linspace(0,iteration)
plt.plot(sumf_list)
plt.yscale('log')
plt.ylim([0.0001, 00])
plt.show()