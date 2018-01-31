#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
The Iris Dataset
=========================================================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length	and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.
"""
print(__doc__)

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets
import numpy as np
from Regression.Logistic.logistic_agent import Agent_harnessing_logistic
from Regression.sq_mean.make_communication import Communication,Circle_communication

n = 10
m = 3
size = int(100/n)
eta = 0.10
iteration = 550

iris = datasets.load_iris()
X1 = iris.data[:100, :(m - 1)]  # we only take the first two features.
One = np.ones([len(X1), 1])
X1 = np.hstack((X1, One))
y = iris.target[:100]

agent_x = [X1[size * i:size * (i + 1)] for i in range(n)]
agent_y = [y[size * i:size * (i + 1)] for i in range(n)]

# Graph = Communication(n, 4, 0.3)
Graph = Circle_communication(n,0.25)
# Graph.make_connected_WS_graph()
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

fig = plt.figure()
# ims= [[] for i in range(n)]
ims = []
Agents = []
for i in range(n):
    Agents.append(Agent_harnessing_logistic(n, m, agent_x[i], agent_y[i], eta, i, Weight_matrix[i]))

for k in range(iteration):
    for i in range(n):
        for j in range(n):
            x_j,name = Agents[i].send(j)
            Agents[j].receive(x_j, name)

    for i in range(n):
        Agents[i].update(k)

    x0 = [np.linspace(0, 10) for i in range(n)]
    # x_ax = [[i for j in range(n)] for i in range(10)]
    x1 = [[]for i in range(n)]
    im = plt.plot()
    for i in range(n):
        x1[i] = (Agents[i].x_i[0] * x0[i] + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
        im += plt.plot(x0[i], x1[i])
    ims.append(im)
    print(k)
for i in range(n):
    print(Agents[i].x_i)


ax = plt.subplot(1,1,1)
p1 = plt.scatter(X1[:50,0],X1[:50,1],color='r',label='Setosa')
p2 = plt.scatter(X1[50:100,0],X1[50:100,1] ,color='b',label='Versicolour')
# plt.plot(x0,x1)
# plt.legend()
legend = ax.legend(handles=[p1,p2],labels=['Setosa','Versicolour'],loc='upper left')
# ani = [None for i in range(n)]
# ani = None
# ani = animation.ArtistAnimation(fig,None)
# for i in range(n):
ani = animation.ArtistAnimation(fig,ims,interval=30)
plt.xlim([0,10])
plt.ylim([0,10])

plt.title('Logistic regression in a part of Iris data set')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
# plt.legend()

# fig.legend()
ani.save(filename='hoge.mp4')
print('save完了')
plt.show()
