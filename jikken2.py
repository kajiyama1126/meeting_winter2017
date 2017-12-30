import numpy as np
import

class Agent(object):
    def __init__(self):
        self.b_i = b_i

        self.x_opt = self.b_i

    def function(self,x):
        return np.dot(x,x)-2*np.dot(self.b_i,x) + np.dot(self.b_i,self.b_i)
