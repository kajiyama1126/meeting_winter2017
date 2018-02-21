import cvxpy as cvx

class Solver(object):
    def __init__(self, n, m, A, b):
        self.n = n
        self.m = m
        self.A = A
        self.b = b

        self.solve()

    def solve(self):
        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(1 / 2 * cvx.power(cvx.norm((self.A[i] * self.x - self.b[i]), 2), 2))
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value)

    def send_opt(self):
        return self.prob.value,self.x.value