import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from progressbar import ProgressBar

from agent.agent import Agent_harnessing, Agent_harnessing_quantize_add_send_data
from Regression.shuron.shuron_jikken1_nonstrongly.shuron_jikken1_nonstrongly_agent import Agent_harnessing_nonstrongly,Agent_harnessing_nonstrongly_quantize_add_send_data
from Regression.shuron.shuron_jikken1_nonstrongly.make_communication import Communication
from Regression.shuron.shuron_jikken1_nonstrongly.problem import Problem

stop_condition = 0.025

class Iteration_multi_nonstrongly(object):
    def __init__(self, n, m, eta,  pattern,count):
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
        self.iterate = 1000000
        self.count =count

    def main(self):
        iterate_count = [[] for i in range(self.pattern)]
        for test in range(self.count):
            print(str(test+1) + '回目')
            self.x_opt, self.f_opt = self.optimal()
            self.P, self.P_history = self.make_communication_graph()
            # f_error_history = [[] for i in range(self.pattern)]
            # send_y_data = [[] for i in range(self.pattern)]
            # send_z_data = [[] for i in range(self.pattern)]
            for i in range(self.pattern):
                iterate_count[i].append(self.iteration(i,stop_condition=stop_condition))
            print(iterate_count)
        return iterate_count


    def optimal(self):  # L1
        """
        :return:  float, float
        """
        self.b = [np.random.rand(self.m) for i in range(self.n)]
        # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
        self.b_num = np.array(self.b)
        # np.reshape(p)
        prob = Problem(self.n, self.m,self.b)
        prob.solve()
        x_opt = np.array(prob.send_x_opt())  # 最適解
        x_opt = np.reshape(x_opt, (-1,))  # reshape
        f_opt = prob.send_f_opt()
        print(x_opt,f_opt)
        return x_opt, f_opt


    # def make_graph(self, f_error):
    #     label = ['Not Quantize', 'Quantize']
    #     line = ['-', '-.']
    #     for i in range(self.pattern):
    #         if i % 2==0:
    #             stepsize = ' $\eta=$' + str(self.eta[i])
    #             plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
    #     for i in range(self.pattern):
    #         if i % 2==1:
    #             stepsize = ' $\eta=$' + str(self.eta[i])
    #             plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
    #
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.xlabel('iteration $k$', fontsize=10)
    #     plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=10)
    #     plt.show()

    # def send_data_check(self,send_y_data,send_z_data):
    #     for i in range(len(send_y_data)):
    #         if i < self.pattern/2:
    #             for j in range(self.n):
    #                 if send_y_data[i][0][j] !=[]:
    #                     plt.plot(send_y_data[i][0][j],'o',label='$||y_{ij}||_{\infty}$')
    #                     plt.plot(send_z_data[i][0][j], 'o', label='$||z_{ij}||_{\infty}$')
    #                     plt.xlabel('iteration $k$')
    #                     plt.legend()
    #                     plt.show()
    #                     break
    #     return

    def make_communication_graph(self):  # 通信グラフを作成＆保存
        weight_graph = Communication(self.n, 6, 0.3)
        weight_graph.make_connected_WS_graph()
        P = weight_graph.P
        P_history = []
        self.sigma = np.linalg.norm(P-1/self.n* np.ones([self.n,self.n]),2)
        print(self.sigma)
        for k in range(self.iterate):  # 通信グラフを作成＆保存
            # weight_graph.make_connected_WS_graph()
            P_history.append(weight_graph.P)
        return P, P_history

    def make_agent(self, pattern):
        Agents = []
        eta = (1-self.sigma)**2/self.eta[pattern]
        I = np.identity(self.m)
        for i in range(self.n):
            if pattern < self.pattern/2:
                Agents.append(Agent_harnessing_nonstrongly(self.n, self.m, I, self.b[i], eta, name=i, weight=None))
            else:
                Agents.append(
                    Agent_harnessing_nonstrongly_quantize_add_send_data(self.n, self.m, I, self.b[i], eta, name=i, weight=None))

        return Agents

    def iteration(self, pattern,stop_condition):
        Agents = self.make_agent(pattern)
        self.f_error_history = []

        f_value = []
        for i in range(self.n):
            state = Agents[i].x_i
            estimate_value = self.optimal_value(state)
            f_value.append(estimate_value)

        error_ini = np.max(f_value) - self.f_opt

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
                state = Agents[i].x_i_hat
                estimate_value = self.optimal_value(state)
                f_value.append(estimate_value)


            # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
            error = np.max(f_value) - self.f_opt
            # if k==0:
            #     error_ini = error
            # print(error)
            # print(error/error_ini)
            if error/error_ini< stop_condition:
                return k

        print('Nostop')
        import sys
        sys.exit()


    def send_f_error_history(self):
        return self.f_error_history

    def send_y_data_zdata(self):
        return self.y_data,self.z_data

    def optimal_value(self, x_i):
        f_opt = 0
        for i in range(self.n):
            for j in range(self.m):
                if abs(x_i[j]-self.b_num[i][j])<=1:
                    f_opt += 1/4 * (x_i[j]-self.b_num[i][j])**4
                else:
                    f_opt += abs(x_i[j]-self.b_num[i][j])-3/4

        return f_opt

class Iteration_multi_nonstrongly_graph(Iteration_multi_nonstrongly):
    def __init__(self, n, m, eta,  pattern,count):
        super(Iteration_multi_nonstrongly_graph, self).__init__(n, m, eta, pattern, count=1)


    def main(self):
        iterate_count = [[] for i in range(self.pattern)]
        iterate_graph = [[] for i in range(self.pattern)]
        iterate_ydata = [[] for i in range(self.pattern)]
        iterate_zdata = [[] for i in range(self.pattern)]
        for test in range(self.count):
            self.x_opt, self.f_opt = self.optimal()
            self.P, self.P_history = self.make_communication_graph()
            # f_error_history = [[] for i in range(self.pattern)]
            # send_y_data = [[] for i in range(self.pattern)]
            # send_z_data = [[] for i in range(self.pattern)]
            for i in range(self.pattern):
                cont , graph,ydata_zdata = self.iteration(i, stop_condition=stop_condition)
                iterate_count[i].append(cont)
                iterate_graph[i] = graph
                if i < self.pattern/2:
                    iterate_ydata[i] = None
                    iterate_zdata[i] = None
                else:
                    iterate_ydata[i]= ydata_zdata[0]
                    iterate_zdata[i]= ydata_zdata[1]


        label = ['[20]', 'Proposed']
        line = ['-', '-.']

        for i in range(self.pattern):
            if i < self.pattern/2:
                stepsize = ' $\eta=$' + str(self.eta[i])
                plt.plot(iterate_graph[i], label=label[0] , linestyle=line[0], linewidth=2)
            else:
                stepsize = ' $\eta=$' + str(self.eta[i])
                plt.plot(iterate_graph[i], label=label[1] , linestyle=line[1], linewidth=2)

        plt.legend()
        plt.yscale('log')
        plt.xlabel('iteration $k$', fontsize=14)
        plt.ylabel('$max_{i}$ $f(\hat{x}_i(k))-f^* / max_{i}$ $f(x_i(0))-f^*$', fontsize=12)

        plt.show()

        for i in range(self.pattern):
            if i >= int(self.pattern/2):
                for j in range(self.n):
                    if iterate_ydata[i][j] !=[]:
                        plt.plot(iterate_ydata[i][j],'o',label='$||y_{ij}||_\infty$',markersize=4)
                        plt.plot(iterate_zdata[i][j],'o',label='$||z_{ij}||_\infty$',markersize=4)
                        plt.xlabel('iteration $k$',fontsize=16)
                        plt.ylabel('$||y_{ij}||_\infty$, $||z_{ij}||_\infty$',fontsize=16)
                        plt.legend()
                        plt.show()
                        break
        return iterate_count

    def iteration(self, pattern,stop_condition):
        Agents = self.make_agent(pattern)
        self.f_error_history = []

        f_value = []
        for i in range(self.n):
            state = Agents[i].x_i
            estimate_value = self.optimal_value(state)
            f_value.append(estimate_value)

        error_ini = np.max(f_value) - self.f_opt

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
                state = Agents[i].x_i_hat
                estimate_value = self.optimal_value(state)
                f_value.append(estimate_value)


            error = np.max(f_value) - self.f_opt
            # if k==0:
            #     error_ini = error
            # print(error)
            self.f_error_history.append(error/error_ini)
            if error/error_ini < stop_condition:
                if pattern >= int(self.pattern/2):
                    # print(Agents[0].send_y_data_zdata())
                    return k,self.f_error_history,Agents[0].send_y_data_zdata()

                return k,self.f_error_history, None

        print('Nostop')
        import sys
        sys.exit()


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
