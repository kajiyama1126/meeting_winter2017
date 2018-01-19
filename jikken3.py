import numpy as np
import copy
import matplotlib.pyplot as plt


def h(k):
    return 0.98**k

def G(k):
    return 0.95**k

def Sum(k):
    sum = 0
    for i in range(k):
        sum += G(k-i)*h(i)
    return sum
# print(Sum(1000))
y = []
z = []
iteration = 10000
for i in range(iteration):
    x = Sum(i)
    y.append(x)
    z.append(x/h(i+1))
plt.plot(y)
plt.yscale('log')
plt.show()
plt.plot(z)
plt.show()