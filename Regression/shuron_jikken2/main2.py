import numpy as np
import progressbar
# from Regression.sq_mean.iteration import Iteration,
from Regression.shuron_jikken1_nonstrongly.iteration import Iteration_multi_nonstrongly,Iteration_multi_nonstrongly_graph
from Regression.shuron_jikken2.make_communication import Circle_communication

np.random.seed(0)  # ランダム値固定
test = False
# test = True
n = 4
m = 2
iteration = 1000000
C_eta = 0.01

alpha = 1
beta = 1

delta_ast = 1.1
delta_v = 1.1
delta_x = 0

mu_x = 0.99997
h_0 = 10
C_h = 0.5

w = 0.0005
d_hat = 2
Graph = Circle_communication(n,w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()
Peron_matrix = Weight_matrix- (1/n)*np.ones([n,n])
sigma = np.linalg.norm(Peron_matrix, 2)

print('sigma',sigma)

eta = 0.0001
G = np.array([[sigma + beta * eta, beta * (beta * eta + 2*d_hat*w), eta * beta * beta], [eta, sigma, 0],
              [0, eta * beta, 1 - alpha * eta]])

l,p = np.linalg.eig(G)
P = np.transpose(p)
P_inv = np.linalg.inv(P)
C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)
print('C_P',C_P)
rho = max(l)
print('mu_x',mu_x)
print('rho',rho)
print('eigen_vec',p)

if l[1] == l[2]:
    C_theta = 3/(rho-l[1])
else:
    C_theta = 1
print('C_theta',C_theta)
# upper_w = np.max(Weight_matrix)

C_1 =C_P * C_theta * (m* n*(delta_v + delta_x + delta_ast))**0.5
Gamma_x = (C_P * C_theta * d_hat * w * (m * n * (beta ** 2 + 1)) ** 0.5) / (mu_x - rho)
Gamma_v = (C_P * C_theta * d_hat * w * (m * n) ** 0.5) / (mu_x - rho)
C_x = (d_hat * w + 2 * eta * beta * m ** 0.5 + eta)
C_v = (d_hat * w * beta * m ** 0.5 + 2 * m * eta * beta ** 2 + d_hat * w + eta * beta * m ** 0.5)
ell_x = C_x * (C_1/(mu_x * h_0) + Gamma_x/(mu_x) + Gamma_v/(C_h*mu_x)) + (2 * d_hat * w + 1) / (2 * mu_x) - 1 / 2
ell_v= C_v * (C_1*C_h/(mu_x * h_0) + C_h*Gamma_x/(mu_x) + Gamma_v/(mu_x)) + (m ** 0.5 * d_hat * beta * w * C_h) / mu_x + (2 * d_hat * w + 1) / (2 * mu_x) - 1 / 2





count = 10

# eta = [0.003,0.005,0.01,0.015,0.02,0.003,0.005,0.01,0.015,0.02]
# eta = [0.01,0.02,0.05,0.01,0.02,0.05]
pattern = len(eta)
print(n, m, count)









if pattern != len(eta):
    print('error')
    pass
else:
    eta = np.reshape(eta, -1)
    #複数回用
    # program = Iteration_multi_nonstrongly(n,m,eta,pattern,count)
    # iteration_count = program.main()

    #一回グラフ作成用
    program = Iteration_multi_nonstrongly_graph(n, m, eta,w, pattern, count,)
    iteration_count = program.main()

    print(np.mean(iteration_count,axis=1))

print('finish2')