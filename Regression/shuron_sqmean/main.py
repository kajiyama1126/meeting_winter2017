import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix
import sys
from Regression.Non_strongly.non_convex_agent import Agent_harnessing_quantize_add_send_data
from agent.agent import Agent_harnessing_quantize_add_send_data_shuron_ex2 as Agent_ex2
from Regression.sq_mean.make_communication import Circle_communication

# test = False
test = True
n = 4
m = 3
iteration = 1000
C_eta = 0.0155

alpha = 1
beta = 1

delta_ast = 3**0.5
delta_v = 2*beta * delta_ast
delta_x = 0

mu_x = 0.98
h_0 = 10
C_h = 0.4

w = 0.05
d_hat = 2
Graph = Circle_communication(n,w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Peron_matrix = Weight_matrix- (1/n)*np.ones([n,n])

sigma1 = np.linalg.norm(Peron_matrix, 2)
sigma = (1-w/(4*n**2))
# eta = alpha *w *C_eta /(4*n**4 * beta**2)
eta = alpha * C_eta /(n**2 * beta**2)*(1-sigma1)
print('eta',eta)
# cc = (2+2**0.5)/4

condition = w/(4*n**2)-w*C_eta/(n**4) * (alpha/beta)**0.5*(5**0.5+2**0.5+1)/8 -2**0.5*w*(d_hat*C_eta)**0.5/n**2
if condition>0:
    print('condition',condition)
else:
    print('error')
    print(condition)
    pass

print('sigma',sigma)
print('sigma1',sigma1)
sigma = sigma1
# rho = max(1 - alpha * eta / 2, sigma + 5 * ((eta * beta) ** 0.5) * ((beta / alpha) ** 0.5))

G = np.array([[sigma + beta * eta, beta * (beta * eta + 2*d_hat*w), eta * beta * beta], [eta, sigma, 0],
              [0, eta * beta, 1 - alpha * eta]])

# G_sym = Matrix(G)
# P, J = G_sym.jordan_form()
# print(P)
# print(J)
l,p = np.linalg.eig(G)
P = np.transpose(p)
P_inv = np.linalg.inv(P)
C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)
print('C_P',C_P)
G_eig,G_eigen_vec = np.linalg.eig(G)
rho = G_eig[0]
print('mu_x',mu_x)
print('rho',rho)
print('eigen_vec',p)

if G_eig[1] == G_eig[2]:
    C_theta = 3/(rho-G_eig[1])
else:
    C_theta = 1
print('C_theta',C_theta)

print(d_hat * w,2*eta*beta,eta)
# upper_w = np.max(Weight_matrix)
upper_w=w
C_1 =C_P * C_theta * (m* n*(delta_v + delta_x + delta_ast))**0.5
Gamma_x = (C_P * C_theta * d_hat * upper_w* (m* n*(beta**2 + 1))**0.5)/(mu_x-rho)
Gamma_v = (C_P * C_theta * d_hat * upper_w*(m* n)**0.5)/(mu_x-rho)
C_x = (d_hat * upper_w + 2 * eta *beta * m ** 0.5 + eta)
C_v = (d_hat * upper_w*beta*m**0.5 + 2 * m *eta*beta**2 + d_hat*upper_w + eta*beta*m**0.5)
ell_x = C_x *(C_1/(mu_x * h_0) + Gamma_x/(mu_x) + Gamma_v/(C_h*mu_x)) + (2*d_hat*upper_w + 1)/(2*mu_x)-1/2
ell_v= C_v *(C_1*C_h/(mu_x * h_0) + C_h*Gamma_x/(mu_x) + Gamma_v/(mu_x)) +(m**0.5 * d_hat *beta * upper_w* C_h)/mu_x + (2*d_hat*upper_w + 1)/(2*mu_x)-1/2


print('ell_x',ell_x,'ell_v',ell_v)
if test is True:
    sys.exit()

A = None
B = []
Agents = []
sumf_list = []
for i in range(n):
    A = np.identity(m)
    b = np.array([-1,0,1]) + 0.05 *np.random.randn(3)
    B.append(b)
    # Agents.append(Agent_harnessing_nonconvex(n, m,A ,b , eta, i, Weight_matrix[i]))
    Agents.append(Agent_ex2(n, m, A, b, eta, i, Weight_matrix[i]))

ave_b = 0
for i in range(n):
    ave_b += B[i] / n

for k in range(iteration):
    for i in range(n):
        for j in range(n):
            x_j, name = Agents[i].send(j)
            Agents[j].receive(x_j, name)

    for i in range(n):
        Agents[i].update(k)

    sumf_list.append(np.linalg.norm(Agents[0].x_i - ave_b))

    if k % 10000 == 0:
        print('iteration',k)


for i in range(n):
    print(Agents[i].x_i)

plt.plot(sumf_list)
plt.yscale('log')
plt.show()

y_data, z_data = Agents[0].send_y_data_zdata()

for i in range(n):
    if y_data[i] == []:
        pass
    else:
        plt.plot(y_data[i], 'o')
        plt.plot(z_data[i], 'o')
        break

plt.show()
