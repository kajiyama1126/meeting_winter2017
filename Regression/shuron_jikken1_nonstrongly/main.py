import numpy as np

# from Regression.sq_mean.iteration import Iteration,
from Regression.shuron_jikken1_nonstrongly.iteration import Iteration_multi_nonstrongly,Iteration_multi_nonstrongly_graph

n = 50
m = 1
np.random.seed(0)  # ランダム値固定

count = 100

# eta = [0.003,0.005,0.01,0.015,0.02,0.003,0.005,0.01,0.015,0.02]
# eta = [0.01,0.02,0.05,0.01,0.02,0.05]
eta = [56,56]
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
    program = Iteration_multi_nonstrongly_graph(n, m, eta, pattern, count)
    iteration_count = program.main()

    print(np.mean(iteration_count,axis=1))

print('finish2')