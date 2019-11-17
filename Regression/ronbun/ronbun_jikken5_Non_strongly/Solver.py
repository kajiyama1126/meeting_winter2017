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
            for j in range(m):
                obj += cvx.Minimize(1 /4 * cvx.power(cvx.abs(self.x[j] - self.b[i][j]), 4))
        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value)

    def send_opt(self):
        return self.prob.value,self.x.value

class Log_Solver(Solver):
    def __init__(self, n, m, A, b,a_i):
        self.n = n
        self.m = m
        self.A = A
        self.b = b
        self.a_i =a_i
        self.solve()

    def solve(self):
        # print(cvx.installed_solvers())
        n, m = self.n, self.m
        self.x = cvx.Variable(m)
        obj = cvx.Minimize(0)
        for i in range(n):
            a = self.a_i[i]
            obj +=  cvx.Minimize(cvx.logistic(a *( self.A[i] *  self.x-self.b[i])))

        self.prob = cvx.Problem(obj)
        self.prob.solve(verbose=True, abstol=1.0e-10, feastol=1.0e-10)
        print(self.prob.status, self.x.value,self.prob.value)
