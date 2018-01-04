import numpy as np
from Regression.Non_strongly.non_convex_agent import Agent_harnessing_nonconvex_quantize_add_send_data,Agent_harnessing_nonconvex
from Regression.sq_mean.make_communication import Communication
import matplotlib.pyplot as plt


n = 10
m =1
iteration = 5000
eta = 0.001

Graph = Communication(n, 4, 0.3)
Graph.make_connected_WS_graph()
Weight_matrix = Graph.send_P()

A =None
B = []
Agents = []
sumf_list = []
for i in range(n):
    b = np.random.rand(1)
    B.append(b)
    # Agents.append(Agent_harnessing_nonconvex(n, m,A ,b , eta, i, Weight_matrix[i]))
    Agents.append(Agent_harnessing_nonconvex_quantize_add_send_data(n, m, A, b, eta, i, Weight_matrix[i]))

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
        if abs(x-B[i])<1:
            sumf+= 1/4*(x-B[i])**4
        else:
            sumf+=abs(x-B[i])-3/4
    sumf_list.append(sumf)


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
plt.show()