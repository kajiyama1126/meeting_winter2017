import numpy as np

from Regression.sq_mean.iteration import Iteration

if __name__ == '__main__':
    n = 25
    m = 20
    np.random.seed(0)  # ランダム値固定

    iteration = 2000

    eta = [0.01,0.01]
    pattern = len(eta)
    print(n, m, iteration)
    if pattern != len(eta):
        print('error')
        pass
    else:
        eta = np.reshape(eta, -1)
        tmp = Iteration(n, m, eta, pattern, iteration)

    print('finish2')