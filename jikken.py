import numpy as np

alpha = 1
beta = 2
eta = 0.000001
lam = 1 - alpha * eta
w = 0.000004

W = np.array([[1 - 2 * w, w, 0, w],
              [w, 1 - 2 * w, w, 0],
              [0, w, 1 - 2 * w, w],
              [w, 0, w, 1 - 2 * w]])

n = 4
m = 4
# I_m = np.identity(m)
# W_eig = np.linalg.norm(W)
# W_kron = np.linalg.norm(np.kron(W,I_m))

AVE = 1 / n * np.ones_like(W)

# EIG = np.linalg.norm(W-AVE)
# EIG_kron = np.linalg.norm(np.kron(W-AVE,I_m),2)
# print(W_eig,EIG,W_kron,EIG_kron)


sigma = np.linalg.norm(W - AVE, 2)
print(sigma)

G = np.array([[sigma + beta * eta, beta * (eta * beta + 2*w), eta * beta * beta],
              [eta, sigma, 0],
              [0, eta * beta, lam]])

G_eig = np.linalg.eigvals(G)
max_G = max(G_eig)

def gamma(G):
    return 1.0/(2*(1-G))


print(max_G)
print(eta * beta * gamma(max_G), eta * gamma(max_G),w*gamma(max_G))
