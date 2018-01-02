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

def log_loss(x,y):
    return math.log(1+math.exp(x))-y*x
def sigmoid(x,y):
    return math.exp(x)/(1+math.exp(x))-y

iris = datasets.load_iris()
X1 = iris.data[:100, :2]  # we only take the first two features.
One = np.ones([len(X1),1])
# print(One)
X1 = np.hstack((X1,One))
# print(One)
y = iris.target[:100]
w = np.ones(3)
eta = 0.2

fig = plt.figure()
ims= []
for iteration in range(2000):
    loss_sum = 0
    for k in range(len(X1)):
        x_data = X1[k]
        y_data = y[k]
        x = np.dot(w,x_data)
        w = w- eta*sigmoid(x,y_data)*x_data

    for k in range(len(X1)):
        x_data = X1[k]
        y_data = y[k]
        x = np.dot(w,x_data)
        loss_sum += log_loss(x,y_data)

    print(w,loss_sum)
    x0 = np.linspace(0, 10)
    x1 = (w[0] * x0 + w[2]) / (-w[1])
    im = plt.plot(x0, x1,color='black')
    ims.append(im)

# print(X1[:,:2
# x0 = np.linspace(0,10)
# x1 = (w[0]*x0 + w[2])/(-w[1])
plt.scatter(X1[:50,0],X1[:50,1],color='r')
plt.scatter(X1[50:100,0],X1[50:100,1] ,color='b')
# plt.plot(x0,x1)
ani = animation.ArtistAnimation(fig,ims,interval=10)
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()
# print(X1)
# print(y)
# data = pd.DataFrame(X, columns=iris.feature_names)
# print(data)