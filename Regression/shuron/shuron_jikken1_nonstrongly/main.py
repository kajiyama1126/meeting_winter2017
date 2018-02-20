import numpy as np

# from Regression.sq_mean.iteration import Iteration,
from Regression.shuron.shuron_jikken1_nonstrongly.iteration import Iteration_multi_nonstrongly,Iteration_multi_nonstrongly_graph

#####################################################################################
#設定パラメータ
n = 10
m = 100
np.random.seed(4)  # ランダム値固定
count = 100
eta = [56] #複数記述可能　[x,y,z,///] eta = (1-\sigma)^2/(x,y,z)
#####################################################################################

eta = eta*2

pattern = len(eta)
print(n, m, count)
if pattern != len(eta):
    print('error')
    pass
else:
    eta = np.reshape(eta, -1)
    #複数回用
    program = Iteration_multi_nonstrongly(n,m,eta,pattern,count)
    iteration_count = program.main()

    #一回グラフ作成用
    # program = Iteration_multi_nonstrongly_graph(n, m, eta, pattern, count)
    # iteration_count = program.main()

    print(np.mean(iteration_count,axis=1))

print('finish2')