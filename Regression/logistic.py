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
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import math
import numpy as np

def log_loss(x,y):
    return math.log(1+math.exp(y*x))-y*x
def sigmoid(x,y):
    return math.exp(x)/(1+math.exp(x))-y

iris = datasets.load_iris()
X1 = iris.data[:100, :]  # we only take the first two features.
y = iris.target[:100]
w = np.ones(4)
eta = 0.01
for k in range(len(X1)):
    x_data = X1[k]
    y_data = y[k]
    x = np.dot(w,x_data)
    w = w - eta*sigmoid(x,y_data)
    print(w)
# print(X1)
# print(y)
# data = pd.DataFrame(X, columns=iris.feature_names)
# print(data)