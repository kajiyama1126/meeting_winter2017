import matplotlib.pyplot as plt
import numpy as np
import progressbar
from progressbar import ProgressBar

from agent.agent import Agent_harnessing,Agent_harnessing_quantize,Agent_harnessing_quantize_add_send_data,Agent_harnessing_grad
from sq_mean.make_communication import Communication
from sq_mean.problem import Problem


class Iteration(object):
    def __init__(self, n, m, eta,  pattern, iterate):
        """
        :param n: int
        :param m: int
        :param lamb: float
        :return: float,float,float
        """
        self.n = n
        self.m = m
        self.eta = eta

        self.pattern = pattern
        self.iterate = iterate

        self.main()

    def optimal(self):  # L1
        """
        :return:  float, float
        """
        self.A = np.array([np.identity(self.m) for i in range(self.n)])
        self.A += 0.1 * np.array([np.random.randn(self.m, self.m) for i in range(self.n)])
        self.b = [np.random.randn(self.m) for i in range(self.n)]
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        self.b_num = np.array(self.b)
        # np.reshape(p)
        prob = Problem(self.n, self.m,self.A,self.b)
        prob.solve()
        x_opt = np.array(prob.x.value)  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        return x_opt, f_opt

    def main(self):
        self.x_opt, self.f_opt = self.optimal()
        print('最適解計算')
        self.P, self.P_history = self.make_communication_graph()
        print('通信グラフ作成')
        f_error_history = [[] for i in range(self.pattern)]
        send_y_data = [[] for i in range(self.pattern)]
        send_z_data = [[] for i in range(self.pattern)]
        for agent in range(self.pattern):
            self.iteration(agent)
            f_error_history[agent] = self.send_f_error_history()
            if agent %2 ==1:
                send_y_data[agent],send_z_data[agent] = self.send_y_data_zdata()
        print('計算終了')
        print('finish')

        self.make_graph(f_error_history)
        self.send_data_check(send_y_data,send_z_data)

    def make_graph(self, f_error):
        label = ['Not Quantize', 'Quantize']
        line = ['-', '-.']
        for i in range(self.pattern):
            if i % 2==0:
                stepsize = ' c=' + str(self.eta[i])
                plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
        for i in range(self.pattern):
            if i % 2==1:
                stepsize = ' c=' + str(self.eta[i])
                plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)

        plt.legend()
        plt.yscale('log')
        plt.xlabel('iteration $k$', fontsize=10)
        plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=10)
        plt.show()

    def send_data_check(self,send_y_data,send_z_data):
        for i in range(len(send_y_data)):
            if i %2 ==1:
                for j in range(self.n):
                    if send_y_data[i][0][j] !=[]:
                        plt.plot(send_y_data[i][0][j],'o',label='$||y_{ij}||_{\infty}$')
                        plt.plot(send_z_data[i][0][j], 'o', label='$||z_{ij}||_{\infty}$')
                        plt.xlabel('iteration $k$')
                        plt.legend()
                        plt.show()
                        break
        return

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 4, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            # weight_graph.make_connected_WS_graph()
            P_history.append(weight_graph.P)
        return P, P_history

    def make_agent(self, pattern):
        Agents = []
        eta = self.eta[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(Agent_harnessing_grad(self.n, self.m,self.A[i], self.b[i],eta, name=i, weight=None))
            elif pattern % 2 == 1:
                Agents.append(
                    Agent_harnessing_quantize_add_send_data(self.n, self.m,self.A[i], self.b[i],eta,  name=i, weight=None))

        return Agents

    def iteration(self, pattern):
        Agents = self.make_agent(pattern)
        self.f_error_history = []
        prog = ProgressBar(max_value=self.iterate)
        for k in range(self.iterate):
            prog.update(k+1)
            # グラフの時間変化
            for i in range(self.n):
                Agents[i].weight = self.P_history[k][i]

            for i in range(self.n):
                for j in range(self.n):
                    state, name = Agents[i].send(j)
                    Agents[j].receive(state, name)
                    # if pattern == 1:
                    #     print(state)
            for i in range(self.n):
                Agents[i].update(k)


            f_value = []
            for i in range(self.n):
                state = Agents[i].x_i
                estimate_value = self.optimal_value(state)
                f_value.append(estimate_value)

            # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
            self.f_error_history.append(np.max(f_value) - self.f_opt)

            if pattern % 2==1:
                self.y_data = [[] for i in range(self.n)]
                self.z_data = [[] for i in range(self.n)]
                for i in range(self.n):
                    self.y_data[i],self.z_data[i] = Agents[i].send_y_data_zdata()

    def send_f_error_history(self):
        return self.f_error_history

    def send_y_data_zdata(self):
        return self.y_data,self.z_data

    def optimal_value(self, x_i):

        b = np.reshape(self.b_num, -1)

        A_tmp = np.reshape(self.A, (-1, self.m))

        tmp = np.dot(A_tmp, np.array(x_i)) - b
        f_opt = 1 / 2 * (np.linalg.norm(tmp, 2)) ** 2
        return f_opt


if __name__ == '__main__':
    n = 50
    m = 50
    lamb = 0.1
    R = 10
    np.random.seed(0)  # ランダム値固定
    pattern = 10
    test = 2000
    step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    # step = [0.25, 0.5, 0.25, 0.7, 0.25, 0.8, 0.25, 0.9, 0.25, 0.95]
    # step = np.array([[0.1 *(j+1) for i in range(2)] for j in range(10)])

    # tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test)
