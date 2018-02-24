
from agent.agent import *
import math
import copy

class Agent_harnessing_quantize_add_send_data_shuron_ronbun_1(Agent_harnessing_quantize_add_send_data):
    def __init__(self, n, m, A, b, weight, name):
        super(Agent_harnessing_quantize_add_send_data_shuron_ronbun_1, self).__init__(n, m, A, b, name, eta)
        self.h_x=h_0
        self.h_v=1./C_h* self.h_x
        self.mu_x = mu_x
        self.send_max_y_data = [[[] for j in range(self.m)]  for i in range(self.n)]
        self.send_max_z_data = [[[] for j in range(self.m)] for i in range(self.n)]

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])
        self.v_i = self.grad()
        self.v = np.zeros([self.n, self.m])

    def make_h(self,k):
        self.h_x = self.h_x *  self.mu_x
        self.h_v = self.h_v *  self.mu_x
        # print(self.h_x)


    def send(self, j):
        if self.weight[j] == 0:
            return None, j
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


class Agent_ADMM_ronbun_2(Agent):
    def __init__(self, n, m, A, b, weight,rho,resol ,name):
        super(Agent_ADMM_ronbun_2, self).__init__(n, m, A, b, weight, name)
        self.rho = rho
        # 量子化の細かさ
        self.q_resol = resol

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        # print(grad)
        return grad

    def grad2(self):
        grad = np.dot(self.A,self.A.T)

        return grad

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.alpha = np.zeros(self.m)
        # self.x = np.zeros([self.n, self.m])
        self.xQ_j = np.zeros([self.n, self.m])
        self.xQ_i = np.zeros([self.n, self.m])

        self.N_i = np.sum(np.sign(self.weight)) - 1
        self.neighbor = np.sign(self.weight)
        self.neighbor[self.name] = 0

    def update_x(self,k):
        sum_x = np.dot(self.neighbor,self.xQ_j)
        x_iQ = self.Quantize(self.x_i)
        part_inv = np.linalg.inv((self.grad2() + 2*self.rho*self.N_i*np.identity(self.m)))
        self.x_i = np.dot(part_inv,(self.rho*self.N_i*x_iQ + self.rho*sum_x-self.alpha))
        # print(self.x_i)

    def update_alpha(self,k):
        x_iQ = self.Quantize(self.x_i)
        sum_x = np.dot(self.neighbor, self.xQ_j)
        self.alpha = self.alpha + self.rho*(self.N_i *x_iQ -sum_x)


    def Quantize(self,x):
        Q_x = np.round(x/self.q_resol)
        # print(Q_x * self.q_resol)
        return Q_x * self.q_resol

    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            self.xQ_i[j] = self.Quantize(self.x_i)
            state = self.xQ_i[j]
            name = j

            return state, name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.xQ_j[name] = x_j

class Agent_YiHong14_ronbun_2(Agent):

    def s(self, k):
        return 10 / (k + 100)

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        return grad

    def __init__(self, n, m, A, b, weight, name):
        super(Agent_YiHong14_ronbun_2, self).__init__(n, m, A, b, weight, name)

        self.Encoder = Yi_Encoder(self.n, self.m)
        self.Decoder = Yi_Decoder(self.n, self.m)
        self.x_E = np.zeros([n, m])
        self.x_D = np.zeros([n, m])
        self.clock = 0
        self.w = self.make_w()
        print(self.w)
        self.send_max_y_data = [[[] for j in range(self.m)]  for i in range(self.n)]

    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x = np.zeros([self.n, self.m])

    def make_w(self):
        weight = copy.copy(self.weight)

        for i in range(len(weight)):
            weight_min = np.min(weight)
            print(weight_min)
            if weight_min > 0:
                return weight_min
            else:
                weight = np.delete(weight, np.argmin(weight))


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

        adjency = np.sign(self.weight)

        self.x_i = self.x_i + self.w*((np.dot(adjency, x)) - self.s(k) * self.grad())
        self.clock += 1

    def send_y_data_zdata(self):
        return self.send_max_y_data, None
class Yi_Encoder(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x_E = np.zeros([n, m])

        self.y = np.zeros([n, m])


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


def sign(x):
    if x >0:
        return 1
    elif x<0:
        return -1
    else:
        return 0


class Agent_Ye(Agent):
    def initial_state(self):
        self.x_i = np.zeros(self.m)
        self.x_hat = np.zeros(self.m)
        self.f_hat = self.grad()
        self.C_a = 1.
        self.C_b = 1
        self.rho = 0.99
        self.bit = 20
        self.tau = 0.1


    def update_x(self,k):
        self.x_i = self.Proj(self.x_i-self.tau*np.dot(self.F_ji,self.nabla))

    def updata_Q_a(self,k):
        self.ell_a = self.C_a * self.rho**k
        self.x_bar_a = self.x_hat
        self.x_hat = self.Q(self.x_i,self.x_bar_a,self.ell_a)

    def updata_Q_b(self,k):
        self.ell_b = self.C_b * self.rho**k
        self.f_bar_b = self.f_hat
        self.f_hat = self.Q(self.grad(),self.f_bar_b,self.ell_b)

    def Q(self,x,bar_x,ell):
        delta = ell/(2**self.bit)
        y = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < bar_x[i]-ell/2:
                y[i] = bar_x[i]-ell/2
            elif x[i] <= bar_x[i]+ell/2 and x[i] >= bar_x[i]-ell/2:
                y[i] = bar_x[i] + sign(x[i]-bar_x[i])*delta*math.floor(abs(x[i]-bar_x[i])/delta)+delta/2
            elif x[i]>bar_x[i]-ell/2:
                y[i] = bar_x[i]+ell/2
            else:
                print('Q_error')

    def Proj(self,x):
        return x

    def grad(self):
        A_to = self.A.T
        grad = np.dot(A_to, (np.dot(self.A, self.x_i) - self.b))
        return grad

#途中
    def send(self, j):
        if self.weight[j] == 0:
            return None, self.name
        else:
            self.xQ_i = self.Quantize(self.x_i)
            state = self.xQ_i
            name = j
            return state, name

    def receive(self, x_j, name):
        if x_j is None:
            pass
        else:
            self.xQ_j[name] = x_j


