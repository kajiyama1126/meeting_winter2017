import numpy as np
import sys

def Condition_proposed(n, m, weight_matrix, w, eta, C_x,C_v, mu, alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0):
    d_hat = np.linalg.norm(np.sign(weight_matrix), 1) - 1

    Peron_matrix = weight_matrix - (1 / n) * np.ones([n, n])
    sigma = np.linalg.norm(Peron_matrix, 2)
    G = np.array([[sigma + beta * eta, beta * (beta * eta + 2 * d_hat * w), eta * beta * beta], [eta, sigma, 0],
                  [0, eta * beta, 1 - alpha * eta]])

    l, p = np.linalg.eig(G)
    rho = max(l)
    P = np.transpose(p)
    P_inv = np.linalg.inv(P)
    C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)

    if mu -rho <= 0:
        print('error')
        print('mu-rho', mu-rho)
        sys.exit()

    if l[1] == l[2]:
        C_theta = 3 / (rho - l[1])
        print('error')
    else:
        C_theta = 1

    C_1 = C_P  * ((delta_v ** 2 + delta_x ** 2 + delta_ast ** 2)) ** 0.5
    Gamma = C_P*d_hat*w*(m*n*(C_x*beta + C_v)**2 +C_x**2)**0.5/(mu-rho)
    Omega_x = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 + 2 * eta * beta * ((1 / n) ** 0.5) + eta)
    Omega_v = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 * beta * (m ** 0.5) + (2 * eta * beta ** 2) * (m / n) ** 0.5 + w * (
             2 * d_hat * (d_hat + 1)) ** 0.5 + eta * beta * (m ** 0.5))
    ell_x = Omega_x/(C_x*mu) * (C_1+ Gamma )+(2*d_hat*w + 1)/(2*mu) -1/2
    ell_v = Omega_v/(C_v*mu) * (C_1+ Gamma )+(m**0.5*d_hat*beta*w*C_x)/(C_v*mu) +(2*d_hat*w + 1)/(2*mu) -1/2
    ell0_x = C_x0/C_x -1/2
    ell0_v = C_v0/C_v -1/2


    print('ell_x0',ell0_x, 'ell_v0',ell0_v)
    print('ell_x', ell_x, 'ell_v', ell_v)

def Condition_find(n, m, weight_matrix, C_x,C_v,  alpha, beta, delta_x, delta_v, delta_ast,C_x0,C_v0):
    import cvxpy as cvx

    w = cvx.Variable()
    eta = cvx.Variable()
    mu = 1
    d_hat = np.linalg.norm(np.sign(weight_matrix), 1) - 1

    Peron_matrix = weight_matrix - (1 / n) * np.ones([n, n])
    sigma = np.linalg.norm(Peron_matrix, 2)
    G = np.array([[sigma + beta * eta, beta * (beta * eta + 2 * d_hat * w), eta * beta * beta], [eta, sigma, 0],
                  [0, eta * beta, 1 - alpha * eta]])

    l, p = np.linalg.eig(G)
    rho = max(l)
    P = np.transpose(p)
    P_inv = np.linalg.inv(P)
    C_P = np.linalg.norm(P, 2) * np.linalg.norm(P_inv, 2)

    if mu -rho <= 0:
        print('error')
        print('mu-rho', mu-rho)
        sys.exit()

    if l[1] == l[2]:
        C_theta = 3 / (rho - l[1])
        print('error')
    else:
        C_theta = 1

    C_1 = C_P  * ((delta_v ** 2 + delta_x ** 2 + delta_ast ** 2)) ** 0.5
    Gamma = C_P*d_hat*w*(m*n*(C_x*beta + C_v)**2 +C_x**2)**0.5/(mu-rho)
    Omega_x = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 + 2 * eta * beta * ((1 / n) ** 0.5) + eta)
    Omega_v = (w * (2 * d_hat * (d_hat + 1)) ** 0.5 * beta * (m ** 0.5) + (2 * eta * beta ** 2) * (m / n) ** 0.5 + w * (
             2 * d_hat * (d_hat + 1)) ** 0.5 + eta * beta * (m ** 0.5))
    ell_x = Omega_x/(C_x*mu) * (C_1+ Gamma )+(2*d_hat*w + 1)/(2*mu) -1/2
    ell_v = Omega_v/(C_v*mu) * (C_1+ Gamma )+(m**0.5*d_hat*beta*w*C_x)/(C_v*mu) +(2*d_hat*w + 1)/(2*mu) -1/2
    ell0_x = C_x0/C_x -1/2
    ell0_v = C_v0/C_v -1/2



def Condition_prior(n, m, laplacian_matrix,w,s_0,gamma,C_x,C_del,C_g,K):
    d_hat = np.max(np.abs(np.diag(laplacian_matrix)))
    print(d_hat)
    l_2, p_2 = np.linalg.eig(laplacian_matrix)
    l_2.sort()
    print(l_2)
    lamb_2 = l_2[1]
    lamb_n = l_2[n - 1]
    rho_h = 1 - w * lamb_2

    tmp1 = (gamma-rho_h)*(rho_h*C_del + w* lamb_n * C_x)/(w*lamb_n)
    tmp2 = C_x/(K+1/2 -w* C_g)

    if w>2./(lamb_2+lamb_n):
        print('error')
        sys.exit()

    if s_0 < max(tmp1,tmp2):
        print('error')
        print(s_0)
        sys.exit()

    if gamma - rho_h < 0:
        print('error')
        print('gamma - rho_h',gamma - rho_h)
        sys.exit()

    M1 = (1 + w * 2 * d_hat) / (2 * gamma) + w * C_g
    M2 = ((m * n) ** 0.5 * w ** 2 * lamb_n * (lamb_n + 2 * gamma * C_g)) / (2 * gamma * (gamma - rho_h))
    M = M1 + M2
    print('M', M)