from quanzize.TAC2019_2019_0620.agent.agent import Agent,Agent_harnessing_quantize_add_send_data
import numpy as np

class Agent_YiHong14(Agent):

    def s(self, k):
        #return 100 / (k + 100)
        return 15 / (k + 100)
        #return 5 / (k + 10)

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        return grad

    def __init__(self, n, m, A, b, weight,w_2, name):
        super(Agent_YiHong14, self).__init__(n, m, A, b, weight, name)
        # self.s = s
        self.Encoder = Yi_Encoder(self.n, self.m)
        self.Decoder = Yi_Decoder(self.n, self.m)
        self.x_E = np.zeros([n, m])
        self.x_D = np.zeros([n, m])
        self.clock = 0
        self.w = w_2
        self.send_max_y_data = [[[] for j in range(self.m)]  for i in range(self.n)]

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])

    # def make_w(self):
    #     count = len(self.weight)
    #
    #     for i in range(count):
    #         weight_min = np.min(self.weight)
    #         if weight_min > 0:
    #             return weight_min
    #         else:
    #             np.delete(self.weight, np.argmin(self.weight))


    def send(self, j):
        if self.weight[j] == 0:
            return None, j
        else:
            self.Encoder.x_encode(self.x_i, j, self.s(self.clock))
            state, name = self.Encoder.send_y_z(j, self.name)
            if self.weight[j] != 0:
                for i in range(self.m):
                    self.send_max_y_data[j][i].append(state[i])
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
        adjency_matrix = -self.weight
        self.x_i = self.x_i + self.w*(np.dot(adjency_matrix, x) - self.s(k) * self.grad())
        self.clock += 1

    def send_y_data_zdata(self):
        return self.send_max_y_data, None
class Yi_Encoder(object):
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


class Yi_Decoder(object):
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


class Agent_harnessing_quantize_add_send_data_ronbun_jikken3(Agent_harnessing_quantize_add_send_data):
    def __init__(self, n, m, A, b,eta, weight, name,C_x,C_v,mu):
        super(Agent_harnessing_quantize_add_send_data_ronbun_jikken3, self).__init__(n, m, A, b, eta,weight,name)
        self.h_x= C_x
        self.h_v= C_v
        self.mu = mu
        self.send_max_y_data = [[[] for j in range(self.m)]  for i in range(self.n)]
        self.send_max_z_data = [[[] for j in range(self.m)] for i in range(self.n)]

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])

    def make_h(self,k):
        self.h_x = self.h_x *  self.mu
        self.h_v = self.h_v *  self.mu
        # print(self.h_x)


    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            self.Encoder.x_encode(self.x_i, j, self.h_x)
            self.Encoder.v_encode(self.v_i, j, self.h_v)
            state, name = self.Encoder.send_y_z(j, self.name)
            if self.weight[j] != 0:
                for i in range(self.m):
                    self.send_max_y_data[j][i].append(state[0][i])
                    self.send_max_z_data[j][i].append(state[1][i])
            # else:
            #     self.send_max_y_data[j].append(None)
            #     self.send_max_z_data[j].append(None)
            return state, name