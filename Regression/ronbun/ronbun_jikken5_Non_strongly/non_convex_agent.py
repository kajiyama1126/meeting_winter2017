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
    def __init__(self, n, m, A, b, eta,weight, name):
        super(Agent_harnessing_nonconvex_quantize_add_send_data, self).__init__(n, m, A, b,eta,weight, name)

    def make_h(self,k):
        self.h_x = 1./(k+1)
        self.h_v = 1.

class Agent_harnessing_nonconvex_quantize_add_send_data_alpha(Agent_harnessing_nonconvex,
                                                        Agent_harnessing_quantize_add_send_data):
    def __init__(self, n, m, A, b, eta, weight, name, alpha):
        self.alpha = alpha
        self.x_0 = np.zeros(m)

        super(Agent_harnessing_nonconvex_quantize_add_send_data_alpha, self).__init__(n, m, A, b, eta, weight, name)
        self.h_x = 0.1
        self.h_v = 0.1



    def grad(self):
        if np.linalg.norm(self.x_i - self.b) <= 1:
            return (self.x_i - self.b) ** 3 + self.alpha * (self.x_i-self.x_0)
        else:
            return np.sign(self.x_i - self.b) + self.alpha * (self.x_i-self.x_0)

    def make_h(self, k):
        self.h_x = self.h_x * 0.95
        self.h_v = self.h_v * 0.95


    def update(self, k):
        self.x_E, self.v_E = self.Encoder.send_x_E_v_E()
        self.x_D, self.v_D = self.Decoder.send_x_D_v_D()
        x = self.x_D - self.x_E
        x[self.name] = np.zeros(self.m)
        v = self.v_D - self.v_E
        v[self.name] = np.zeros(self.m)

        grad_bf = self.grad()
        self.x_i = self.x_i + np.dot(self.weight, x) - self.eta * self.v_i
        self.v_i = self.v_i + np.dot(self.weight, v) + (self.grad() - grad_bf)

        self.make_h(k)
        # if k%10 == 0:
        #     self.x_0 = self.x_i

class Agent_harnessing_nonconvex_quantize_add_send_data_alpha2(Agent_harnessing_nonconvex_quantize_add_send_data_alpha):

    def __init__(self, n, m, A, b, eta, weight, name, alpha,a_i):
        self.alpha = alpha
        self.x_0 = np.zeros(m)
        self.a_i = a_i

        super(Agent_harnessing_nonconvex_quantize_add_send_data_alpha, self).__init__(n, m, A, b, eta, weight, name)
        self.h_x = 0.1
        self.h_v = 0.1

    def initial_state(self):
        self.x_i = np.random.rand(self.m)
        # self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])


    def grad(self):

        return self.a_i * (math.e**(self.a_i*(np.dot(self.A,self.x_i)-self.b)))/(1+math.e**(self.a_i*(np.dot(self.A,self.x_i)-self.b))) * self.A+ self.alpha*self.x_i

    def make_h(self, k):
        self.h_x = self.h_x * 0.95
        self.h_v = self.h_v * 0.95

        # if np.linalg.norm(self.x_i - self.b) <= 1:
        #     return (self.x_i - self.b) ** 3 + self.alpha * (self.x_i-self.x_0)
        # else:
        #     return np.sign(self.x_i - self.b) + self.alpha * (self.x_i-self.x_0)
