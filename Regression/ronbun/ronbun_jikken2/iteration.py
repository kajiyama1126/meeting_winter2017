import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar

from Regression.ronbun.ronbun_jikken2.make_communication import Communication
from Regression.ronbun.ronbun_jikken2.problem import Problem
from Regression.ronbun.ronbun_jikken2.ronbun_agent2 import Agent_YiHong14_ronbun_2,Agent_ADMM_ronbun_2,Agent_DA_ronbun2,Agent_DA_Quantize_ronbun2
from agent.agent import Agent_harnessing, Agent_harnessing_quantize_add_send_data


class Iteration_multi(object):
    def __init__(self, n, m, parameter,  algo, count,stop_condition):
        """
        :param n: int
        :param m: int
        :param lamb: float
        :return: float,float,float
        """
        self.n = n
        self.m = m

        self.parameter = parameter

        self.pattern = len(algo)
        self.iterate = 10000
        self.algo = algo
        self.count = count
        self.stop_condition = stop_condition

    def main(self):
        iterate_count = [[] for i in range(self.pattern)]
        for test in range(self.count):
            print(str(test + 1) + '回目')
            self.x_opt, self.f_opt = self.optimal()
            self.P, self.P_history = self.make_communication_graph()

            for i in range(self.pattern):
                iterate_count[i].append(self.iteration(i, self.stop_condition))

        return iterate_count

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
        prob = Problem(self.n, self.m, self.A, self.b)
        prob.solve()
        x_opt = np.array(prob.x.value)  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        return x_opt, f_opt

    # def send_data_check(self, send_y_data, send_z_data):
    #     for i in range(len(send_y_data)):
    #         if i < self.pattern / 2:
    #             for j in range(self.n):
    #                 if send_y_data[i][0][j] != []:
    #                     plt.plot(send_y_data[i][0][j], 'o', label='$||y_{ij}||_{\infty}$')
    #                     plt.plot(send_z_data[i][0][j], 'o', label='$||z_{ij}||_{\infty}$')
    #                     plt.xlabel('iteration $k$')
    #                     plt.legend()
    #                     plt.show()
    #                     break
    #     return

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 4, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            # weight_graph.make_connected_WS_graph()#グラフの時間変化
            P_history.append(weight_graph.P)
        return P, P_history

    def make_agent(self, pattern):
        Agents = []
        param = self.parameter[pattern]
        algo = self.algo[pattern]
        for i in range(self.n):
            if algo == 'Harnessing':
                eta = param
                Agents.append(Agent_harnessing(self.n, self.m, self.A[i], self.b[i], eta,self.P[i],  name=i))
            elif algo == 'Proposed':
                eta = param
                Agents.append(
                    Agent_harnessing_quantize_add_send_data(self.n, self.m, self.A[i], self.b[i],eta, self.P[i],
                                                            name=i))
            elif algo == 'Subgrad':
                Agents.append(Agent_YiHong14_ronbun_2(self.n, self.m, self.A[i], self.b[i], self.P[i], name=i))
            elif algo == 'ADMM':
                rho = param[0]
                resol = param[1]
                Agents.append(Agent_ADMM_ronbun_2(self.n, self.m, self.A[i], self.b[i], self.P[i],rho,resol, name=i))
            elif algo == 'DA':
                delta = param
                Agents.append(Agent_DA_Quantize_ronbun2(self.n, self.m, self.A[i], self.b[i], self.P[i],delta, name=i))
            # elif algo == 'stochastic':
            #     Agents.append(Agent_harnessing_quantize_add_send_data(self.n, self.m, self.A[i], self.b[i], eta, name=i, weight=None))

        return Agents

    def iteration(self, pattern, stop_condition):
        self.Agents = self.make_agent(pattern)
        self.f_error_history = []
        prog = ProgressBar(max_value=self.iterate)
        for k in range(self.iterate):
            prog.update(k + 1)
            # グラフの時間変化
            for i in range(self.n):
                self.Agents[i].weight = self.P_history[k][i]

            if self.algo[pattern] == 'Harnessing' or self.algo[pattern] == 'Proposed' or self.algo[pattern] == 'Subgrad' or self.algo[pattern] == 'DA':
                self.pro_update(k)
            elif self.algo[pattern] == 'ADMM':
                self.ADMM_update(k)


            f_value = []
            for i in range(self.n):
                state = self.Agents[i].send_estimate()
                estimate_value = self.optimal_value(state)
                f_value.append(estimate_value)

            # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
            error = np.max(f_value) - self.f_opt
            print(error)
            if error < stop_condition:
                return k


        print('Nostop')
        return k
        import sys
        sys.exit()

    def pro_update(self,k):
        for i in range(self.n):
            for j in range(self.n):
                state, name = self.Agents[i].send(j)
                self.Agents[j].receive(state, name)
                # if pattern == 1:
        for i in range(self.n):
            self.Agents[i].update(k)


    def ADMM_update(self, k):
        for i in range(self.n):
            self.Agents[i].update_x(k)

        for i in range(self.n):
            for j in range(self.n):
                x_j, name = self.Agents[i].send(j)
                self.Agents[j].receive(x_j, name)

        for i in range(self.n):
            self.Agents[i].update_alpha(k)

    def send_f_error_history(self):
        return self.f_error_history

    def send_y_data_zdata(self):
        return self.y_data, self.z_data

    def optimal_value(self, x_i):

        b = np.reshape(self.b_num, -1)

        A_tmp = np.reshape(self.A, (-1, self.m))

        tmp = np.dot(A_tmp, np.array(x_i)) - b
        f_opt = 1 / 2 * (np.linalg.norm(tmp, 2)) ** 2
        return f_opt

    # class Iteration_multi_graph(Iteration_multi):
    #     def __init__(self, n, m, eta, pattern, count):
    #         super(Iteration_multi_graph, self).__init__(n, m, eta, pattern, count=1)
    #
    #     def main(self):
    #         iterate_count = [[] for i in range(self.pattern)]
    #         iterate_graph = [[] for i in range(self.pattern)]
    #         iterate_ydata = [[] for i in range(self.pattern)]
    #         iterate_zdata = [[] for i in range(self.pattern)]
    #         for test in range(self.count):
    #             self.x_opt, self.f_opt = self.optimal()
    #             self.P, self.P_history = self.make_communication_graph()
    #             # f_error_history = [[] for i in range(self.pattern)]
    #             # send_y_data = [[] for i in range(self.pattern)]
    #             # send_z_data = [[] for i in range(self.pattern)]
    #             for i in range(self.pattern):
    #                 cont, graph, ydata_zdata = self.iteration(i, stop_condition=0.0001)
    #                 iterate_count[i].append(cont)
    #                 iterate_graph[i] = graph
    #                 if i < self.pattern / 2:
    #                     iterate_ydata[i] = None
    #                     iterate_zdata[i] = None
    #                 else:
    #                     iterate_ydata[i] = ydata_zdata[0]
    #                     iterate_zdata[i] = ydata_zdata[1]
    #
    #         label = ['[20]', 'Proposed']
    #         line = ['-', '-.']
    #
    #         for i in range(self.pattern):
    #             if i < self.pattern / 2:
    #                 stepsize = ' $\eta=$' + str(self.eta[i])
    #
    #                 plt.plot(iterate_graph[i], label=label[0] + stepsize, linestyle=line[0], linewidth=2)
    #             else:
    #                 stepsize = ' $\eta=$' + str(self.eta[i])
    #                 plt.plot(iterate_graph[i], label=label[1] + stepsize, linestyle=line[1], linewidth=2)
    #
    #         plt.legend()
    #         plt.yscale('log')
    #         plt.grid(which='major', color='black', linestyle='-')
    #         plt.grid(which='minor', color='gray', linestyle=':')
    #         plt.xlabel('iteration $k$', fontsize=16)
    #         plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=16)
    #         plt.show()
    #
    #         for i in range(self.pattern):
    #             if i >= int(self.pattern / 2):
    #                 for j in range(self.n):
    #                     if iterate_ydata[i][j] != []:
    #                         plt.plot(iterate_ydata[i][j], 'o', label='$||y_{ij}||_\infty$', markersize=4)
    #                         plt.plot(iterate_zdata[i][j], 'o', label='$||z_{ij}||_\infty$', markersize=4)
    #                         plt.xlabel('iteration $k$', fontsize=16)
    #                         plt.ylabel('$||y_{ij}||_\infty$, $||z_{ij}||_\infty$', fontsize=16)
    #                         plt.legend()
    #                         plt.show()
    #                         break
    #         return iterate_count
    #
    #     def iteration(self, pattern, stop_condition):
    #         Agents = self.make_agent(pattern)
    #         self.f_error_history = []
    #         prog = ProgressBar(max_value=self.iterate)
    #         for k in range(self.iterate):
    #             prog.update(k + 1)
    #             # グラフの時間変化
    #             for i in range(self.n):
    #                 Agents[i].weight = self.P_history[k][i]
    #
    #             for i in range(self.n):
    #                 for j in range(self.n):
    #                     state, name = Agents[i].send(j)
    #                     Agents[j].receive(state, name)
    #                     # if pattern == 1:
    #                     #     print(state)
    #             for i in range(self.n):
    #                 Agents[i].update(k)
    #
    #             f_value = []
    #             for i in range(self.n):
    #                 state = Agents[i].x_i
    #                 estimate_value = self.optimal_value(state)
    #                 f_value.append(estimate_value)
    #
    #             self.f_error_history.append(np.max(f_value) - self.f_opt)
    #             error = np.max(f_value) - self.f_opt
    #             if error < stop_condition:
    #                 if pattern >= int(self.pattern / 2):
    #                     # print(Agents[0].send_y_data_zdata())
    #                     return k, self.f_error_history, Agents[0].send_y_data_zdata()
    #
    #                 return k, self.f_error_history, None
    #
    #         print('Nostop')
    #         import sys
    #         sys.exit()

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
