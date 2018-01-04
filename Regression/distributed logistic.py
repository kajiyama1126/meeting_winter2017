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
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import math
import numpy as np
from agent.agent import Agent_harnessing,Agent_harnessing_logistic
from sq_mean.make_communication import Communication

n = 10
m = 3
size = int(100/n)
eta = 0.10
iteration = 1000

iris = datasets.load_iris()
X1 = iris.data[:100, :(m - 1)]  # we only take the first two features.
One = np.ones([len(X1), 1])
X1 = np.hstack((X1, One))
y = iris.target[:100]

agent_x = [X1[size * i:size * (i + 1)] for i in range(n)]
agent_y = [y[size * i:size * (i + 1)] for i in range(n)]

Graph = Communication(n, 4, 0.3)
Graph.make_connected_WS_graph()
Weight_matrix = Graph.send_P()

fig = plt.figure()
ims= [[] for i in range(n)]

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

    for i in range(n):
        x0 = np.linspace(0, 10)
        x1 = (Agents[i].x_i[0] * x0 + Agents[i].x_i[2]) / (-Agents[i].x_i[1])
        im = plt.plot(x0, x1)
        ims[i].append(im)
    print(k)
for i in range(n):
    print(Agents[i].x_i)



plt.scatter(X1[:50,0],X1[:50,1],color='r')
plt.scatter(X1[50:100,0],X1[50:100,1] ,color='b')
# plt.plot(x0,x1)
ani = [None for i in range(n)]
for i in range(n):
    ani[i] = animation.ArtistAnimation(fig,ims[i],interval=30)
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()