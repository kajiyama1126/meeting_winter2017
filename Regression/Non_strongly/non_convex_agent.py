from agent.agent import Agent_harnessing_quantize_add_send_data,Agent_harnessing
import math
import numpy as np

class  Agent_harnessing_nonconvex(Agent_harnessing):

    def grad(self):
        if np.linalg.norm(self.x_i-self.b) <= 1:
            return (self.x_i-self.b) ** 3
        else:
            return np.sign(self.x_i-self.b)

class Agent_harnessing_nonconvex_quantize_add_send_data(Agent_harnessing_nonconvex,Agent_harnessing_quantize_add_send_data):
    def __init__(self,n,m,A,b,eta,name,weight):
        super(Agent_harnessing_nonconvex_quantize_add_send_data, self).__init__(n, m, A, b, eta, name, weight)

    def make_h(self,k):
        self.h_x = 1./(k+1)
        self.h_v = 1.
        # self.h_v = 0.1/(k+1)