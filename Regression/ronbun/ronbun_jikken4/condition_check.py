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

    print('muy-rho',mu-rho)
    print('ell_x0',ell0_x, 'ell_v0',ell0_v)
    print('ell_x', ell_x, 'ell_v', ell_v)


if __name__ == '__main__':
    # 共通parameter============================================================
    patterns = 1
    iteration = 400000
    n = 4
    m = 2
    K = 1  # 先行研究でのellに相当(量子化レベル)
    #####################################################################################
    # f_i = ||Ax-b||_2^2の場合
    A1 = np.array([[0.7, 0.2], [0.3, 0.6]])
    # A1 = np.array([[1, 2], [1, 1]])
    # b = np.array([-1.5, 0.5]) + 1.0 * np.random.rand(2)

    alpha = np.linalg.norm(np.dot(A1.T, A1), 2)
    beta = np.linalg.norm(np.dot(A1.T, A1), 2)
    print(alpha, beta)
    C_g = np.linalg.norm(np.dot(np.dot(A1.T, A1), np.array([-1, 1])), 2)
    #####################################################################################