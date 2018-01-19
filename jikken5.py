import numpy as np
import matplotlib.pyplot as plt

def Gamma(w):
    return w**2/((1-(1-w/(2*4))**0.5)**2)

print(Gamma(0.1))
for i in range(10000):
    k = (i+1)/10000
    print(Gamma(k))