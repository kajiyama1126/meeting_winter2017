from agent.agent import Agent,Agent_harnessing_quantize_add_send_data
import numpy as np

class Agent_YiHong14(Agent):

    def s(self, k):
        return 1 / (k + 1)

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        return grad

    def __init__(self, n, m, A, b, eta, name, weight):
        super(Agent_YiHong14, self).__init__(n, m, A, b, eta, name, weight)
        # self.s = s
        self.Encoder = Encoder(self.n, self.m)
        self.Decoder = Decoder(self.n, self.m)
        self.x_E = np.zeros([n, m])
        self.x_D = np.zeros([n, m])
        self.clock = 0
        self.w = self.make_w()

    def make_w(self):
        count = len(self.weight)

        for i in range(count):
            weight_min = np.min(self.weight)
            if weight_min > 0:
                return weight_min
            else:
                np.delete(self.weight, np.argmin(self.weight))

    def send(self, j):
        if self.weight[j] == 0:
            return None, j
        else:
            self.Encoder.x_encode(self.x_i, j, self.s(self.clock))
            state, name = self.Encoder.send_y_z(j, self.name)
            return state, name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.Decoder.get_y_z(x_j, name)
            self.Decoder.x_decode(name, self.s(self.clock))

    def update(self, k):
        self.x_E = self.Encoder.send_x_E_v_E()
        self.x_D = self.Decoder.send_x_D_v_D()
        x = self.x_D - self.x_E
        x[self.name] = np.zeros(self.m)

        self.x_i = self.x_i + self.w * (np.dot(self.weight, x)) - self.s(k) * self.grad()
        self.clock += 1


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

    def send_y_z(self, j, name):
        return self.y[j], name

    def send_x_E_v_E(self):
        # print(self.x_E, self.v_E)
        return self.x_E

    def send_x_v(self, name):
        return self.x_E[name]

    def quantize(self, x_i):
        tmp = np.around(x_i)
        # print(max(tmp))
        return tmp


class Decoder(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x_D = np.zeros([n, m])
        self.y = np.zeros([n, m])

    def get_y_z(self, state, j):
        self.y[j] = state

    def x_decode(self, j, h_x):
        self.x_D[j] = h_x * self.y[j] + self.x_D[j]

    def send_x_v(self, name):
        return self.x_D[name]

    def send_x_D_v_D(self):
        return self.x_D


class Agent_harnessing_quantize_add_send_data_shuron_jikken2(Agent_harnessing_quantize_add_send_data):
    def __init__(self,n,m,A,b,eta,name,weight):
        super(Agent_harnessing_quantize_add_send_data_shuron_jikken2, self).__init__(n, m, A, b, eta, name, weight)
        self.h_x=10
        self.h_v=1/0.4 * self.h_x

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])

    def make_h(self,k):
        self.h_x = self.h_x *  0.9999
        self.h_v = self.h_v *  0.9999
        # print(self.h_x)