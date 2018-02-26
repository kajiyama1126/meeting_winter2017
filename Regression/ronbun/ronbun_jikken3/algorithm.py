from progressbar import ProgressBar

from Regression.ronbun.ronbun_jikken2.ronbun_agent2 import *


class Distributed_solver(object):
    def __init__(self, n, m, A, b, W_matrix, algo, iteration,other_param):
        self.n = n
        self.m = m
        self.A = A
        self.b = b

        self.weight_matrix = W_matrix
        self.algo = algo
        self.iteration = iteration

        #エージェントごとに必要なパラメータを設定(タプル)
        self.other_param = other_param

    def Make_agent(self):
        self.Agents = []
        for i in range(self.n):
            if self.algo == 'Proposed':
                eta = self.other_param[0]
                mu_x = self.other_param[1]
                C_h = self.other_param[2]
                h_0 = self.other_param[3]
                agent = Agent_harnessing_quantize_add_send_data_shuron_ronbun_2(self.n, self.m, self.A[i], self.b[i], i,
                                                                                eta)
            elif self.algo == 'ADMM':
                rho = self.other_param[0]
                resol = self.other_param[1]
                agent = Agent_ADMM_ronbun_2(self.n, self.m, self.A[i], self.b, self.weight_matrix[i], i)
            elif self.algo == 'Subgrad':
                agent = Agent_YiHong14_ronbun_2(self.n, self.m, self.A[i], self.b[i], self.weight_matrix[i], i)

            self.Agents.append(agent)

    def Iteration(self,f_opt):
        self.f_opt = f_opt
        self.prog = ProgressBar(max_value=self.iteration)
        if self.algo == 'Proposed':
            result = self.Proposed_algo()
        elif self.algo == 'Subgrad':
            result = self.Subgrad_algo()
        elif self.algo == 'ADMM':
            result = self.ADMM_algo()

        return result

    def Proposed_algo(self):
        error_data = []
        for k in range(self.iteration):
            self.prog.update(k+1)

            for i in range(self.n):
                for j in range(self.n):
                    x_j, name = self.Agents[i].send(j)
                    self.Agents[j].receive(x_j, name)

            for i in range(self.n):
                self.Agents[i].update(k)

            error = self.error_check()
            error_data.append(error)
        return error_data

    def Subgrad_algo(self):
        error_data = []
        for k in range(self.iteration):
            self.prog.update(k + 1)

            for i in range(self.n):
                for j in range(self.n):
                    x_j, name = self.Agents[i].send(j)
                    self.Agents[j].receive(x_j, name)

            for i in range(self.n):
                self.Agents[i].update(k)

            error = self.error_check()
            error_data.append(error)
        return error_data

    def ADMM_algo(self):
        error_data = []
        for k in range(self.iteration):
            self.prog.update(k + 1)

            for i in range(self.n):
                self.Agents[i].update_x(k)

            for i in range(self.n):
                for j in range(self.n):
                    x_j, name = self.Agents[i].send(j)
                    self.Agents[j].receive(x_j, name)

            for i in range(self.n):
                self.Agents[i].update_alpha(k)

            error = self.error_check()
            error_data.append(error)
        return error_data

    def error_check(self):
        f_value = []
        for i in range(self.n):
            state = self.Agents[i].x_i
            estimate_value = self.optimal_value(state)
            f_value.append(estimate_value)

        return np.max(f_value) - self.f_opt

    def optimal_value(self,x):
        f = 0
        for i in range(self.n):
            f += 1/2 * np.linalg.norm(np.dot(self.A[i],x)-self.b[i])**2

        return f
