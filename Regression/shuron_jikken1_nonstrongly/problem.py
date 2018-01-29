import cvxpy as cvx
import numpy as np


class Problem(object):
    def __init__(self, n, m, b):
        self.n = n
        self.m = m
        self.A = [np.identity(self.m) for i in range(self.n)]
        self.b = b


    def solve(self):
        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(1 / 4 * cvx.power(cvx.norm((self.A[i]*self.x - self.b[i]), 2), 4))
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=False,abstol=1.0e-10,feastol=1.0e-10)
        print(self.prob.status, self.x.value)


    def send_x_opt(self):
        return self.x.value

    def send_f_opt(self):
        return self.prob.value



