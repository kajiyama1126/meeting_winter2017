import numpy as np
import math

class Agent(object):
    def __init__(self, n, m, A, b, weight, name):
        self.n = n
        self.m = m
        self.A = A
        self.b = b
        self.name = name
        self.weight = weight

        self.initial_state()

    def initial_state(self):
        self.x_i = np.random.rand(self.m)
        self.x = np.zeros([self.n, self.m])

    def send(self, j):
        return self.x_i, self.name

    def receive(self, x_j, name):
        self.x[name] = x_j

    def send_estimate(self):
        return self.x_i

class Agent_harnessing(Agent):
    def __init__(self, n, m, A, b, eta, weight, name):
        super(Agent_harnessing, self).__init__(n, m, A, b, weight, name)
        self.eta = eta

    def initial_state(self):
        self.x_i = np.random.rand(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        return grad

    def send(self, j):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i
        grad_bf = self.grad()
        self.x_i = np.dot(self.weight, self.x) - self.eta * self.v_i
        self.v_i = np.dot(self.weight, self.v) + (self.grad() - grad_bf)


# class Agent_harnessing_grad(Agent_harnessing):
#     def func(self,x):
#         return np.linalg.norm(np.dot(self.A,x)-self.b)**2
#
#     def local_opt(self):




class Agent_harnessing_quantize(Agent_harnessing):
    def __init__(self, n, m, A, b,eta, weight, name):
        super(Agent_harnessing_quantize, self).__init__(n, m, A, b, eta, weight, name)
        self.eta = eta
        self.Encoder = Encoder(self.n, self.m)
        self.Decoder = Decoder(self.n, self.m)
        self.x_E = np.zeros([n, m])
        self.v_E = np.zeros([n, m])
        self.x_D = np.zeros([n, m])
        self.v_D = np.zeros([n, m])
        self.h_x = 1
        self.h_v = 1

    def make_h(self,k):
        self.h_x = self.h_x * 0.99
        self.h_v = self.h_v * 0.99


    def send(self, j):
        if self.weight[j] == 0:
            return None, j
        else:
            self.Encoder.x_encode(self.x_i, j, self.h_x)
            self.Encoder.v_encode(self.v_i, j, self.h_v)
            state, name = self.Encoder.send_y_z(j, self.name)
            return state, name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.Decoder.get_y_z(x_j, name)
            self.Decoder.x_decode(name, self.h_x)
            self.Decoder.v_decode(name, self.h_v)

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
        # print(self.h_x,self.h_v)
        # print(self.x_D,self.x_E)


class Encoder(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x_E = np.zeros([n, m])
        self.v_E = np.zeros([n, m])

        self.y = np.zeros([n, m])
        self.z = np.zeros([n, m])

    def x_encode(self, x_i, j, h_x):
        tmp = (x_i - self.x_E[j]) / h_x
        # print(tmp,x_i - self.x_E[j],h_x)
        self.y[j] = self.quantize(tmp)
        self.x_E[j] = h_x * self.y[j] + self.x_E[j]

    def v_encode(self, v_i, j, h_v):
        tmp = (v_i - self.v_E[j]) / h_v
        self.z[j] = self.quantize(tmp)
        self.v_E[j] = h_v * self.z[j] + self.v_E[j]

    def send_y_z(self, j, name):
        return (self.y[j], self.z[j]), name

    def send_x_E_v_E(self):
        # print(self.x_E, self.v_E)
        return self.x_E, self.v_E

    def send_x_v(self, name):
        return self.x_E[name], self.v_E[name]

    def quantize(self, x_i):
        tmp = np.around(x_i)
        # print(max(tmp))
        return tmp


class Decoder(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x_D = np.zeros([n, m])
        self.v_D = np.zeros([n, m])

        self.y = np.zeros([n, m])
        self.z = np.zeros([n, m])

    def get_y_z(self, state, j):
        self.y[j] = state[0]
        self.z[j] = state[1]

    def x_decode(self, j, h_x):
        self.x_D[j] = h_x * self.y[j] + self.x_D[j]

    def v_decode(self, j, h_v):
        self.v_D[j] = h_v * self.z[j] + self.v_D[j]

    def send_x_v(self, name):
        return self.x_D[name], self.v_D[name]

    def send_x_D_v_D(self):
        return self.x_D, self.v_D


class Agent_harnessing_quantize_add_send_data(Agent_harnessing_quantize):
    def __init__(self, n, m, A, b, eta, weight, name):
        super(Agent_harnessing_quantize_add_send_data, self).__init__(n, m, A, b, eta,weight,name)
        self.send_max_y_data = [[] for i in range(self.n)]
        self.send_max_z_data = [[] for i in range(self.n)]

    def send(self, j):
        if self.weight[j] == 0:
            return None, j
        else:
            self.Encoder.x_encode(self.x_i, j, self.h_x)
            self.Encoder.v_encode(self.v_i, j, self.h_v)
            state, name = self.Encoder.send_y_z(j, self.name)
            if self.weight[j] != 0:
                self.send_max_y_data[j].append(np.linalg.norm(state[0], np.inf))
                self.send_max_z_data[j].append(np.linalg.norm(state[1], np.inf))
            # else:
            #     self.send_max_y_data[j].append(None)
            #     self.send_max_z_data[j].append(None)
            return state, name

    def send_y_data_zdata(self):
        return self.send_max_y_data, self.send_max_z_data



