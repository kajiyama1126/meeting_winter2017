import matplotlib.pyplot as plt
import numpy as np

from agent.agent import Agent_harnessing,Agent_harnessing_quantize
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
        for agent in range(self.pattern):
            f_error_history[agent] = self.iteration(agent)
        print('計算終了')
        print('finish')

        self.make_graph(f_error_history)

    def make_graph(self, f_error):
        label = ['DSM', 'Proposed']
        line = ['-', '-.']
        for i in range(self.pattern):
            # stepsize = '_s(k)=' + str(self.step[i]) + '/k+10'
            stepsize = ' c=' + str(self.eta[i])
            plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
        plt.legend()
        plt.yscale('log')
        plt.xlabel('iteration $k$', fontsize=10)
        plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=10)
        plt.show()

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
                Agents.append(Agent_harnessing(self.n, self.m,self.A[i], self.b[i], name=i, weight=None))
            elif pattern % 2 == 1:
                Agents.append(
                    Agent_harnessing_quantize(self.n, self.m,self.A[i], self.b[i],  name=i, weight=None))

        return Agents

    def iteration(self, pattern):
        Agents = self.make_agent(pattern)
        f_error_history = []
        for k in range(self.iterate):
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
            f_error_history.append(np.max(f_value) - self.f_opt)

        return f_error_history

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
