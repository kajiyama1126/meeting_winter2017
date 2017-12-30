import numpy as np
import copy
J = np.array([[0.8,1,0],
             [0,0.85,1],
             [0,0,0.9]])

J1 = copy.copy(J)
n=200
for i in range(n):
    J1 = np.dot(J1,J)
    print(J1)
