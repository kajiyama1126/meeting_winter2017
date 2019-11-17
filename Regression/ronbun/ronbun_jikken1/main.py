import numpy as np


#from Regression.ronbun.ronbun_jikken1.iteration import Iteration_multi_graph,Iteration_multi
from ronbun.ronbun_jikken1.iteration import Iteration_multi_graph, Iteration_multi

#####################################################################################
#設定パラメータ
n = 100
m = 4
#n = 25
#m = 20
np.random.seed(0)  # ランダム値固定

count = 100 #平均取る回数
#eta = [0.003, 0.005, 0.01, 0.015, 0.02]
eta = [0.02, 0.03, 0.05, 0.1]
#####################################################################################


eta = eta * 2

pattern = len(eta)
print(n, m, count)
if pattern != len(eta):
    print('error')
    pass
else:
    eta = np.reshape(eta, -1)
    #複数回用
    #program = Iteration_multi(n,m,eta,pattern,count)
    #iteration_count = program.main()

    #一回グラフ作成用
    program = Iteration_multi_graph(n, m, eta, pattern, count)
    iteration_count = program.main()

    print(np.mean(iteration_count, axis=1))

print('finish2')