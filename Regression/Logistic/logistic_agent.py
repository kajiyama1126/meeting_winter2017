from agent.agent import Agent_harnessing_quantize_add_send_data,Agent_harnessing
import math
import numpy as np

class Agent_harnessing_logistic(Agent_harnessing):

    def grad(self):
        g = 0
        # print(len(self.A))
        for i in range(len(self.A)):
            x = np.dot(self.x_i,self.A[i])
            g += ((math.exp(x) / (1 + math.exp(x))) - self.b[i])*self.A[i]

        return g

class Agent_harnessing_logistic_quantize_add_send_data(Agent_harnessing_logistic,Agent_harnessing_quantize_add_send_data):
    def __init__(self, n, m, A, b, weight, name):
        super(Agent_harnessing_logistic_quantize_add_send_data, self).__init__(n, m, A, b, name, eta)