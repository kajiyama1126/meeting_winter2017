import numpy as np


from Regression.shuron.shuron_jikken1.iteration import Iteration_multi,Iteration_multi_graph

#####################################################################################
#設定パラメータ
n = 25
m = 20
np.random.seed(0)  # ランダム値固定

count = 100
eta = [0.003,0.005,0.01,0.015,0.02]
#####################################################################################

eta = eta *2
pattern = len(eta)
print(n, m, count)
if pattern != len(eta):
    print('error')
    pass
else:
    eta = np.reshape(eta, -1)
    #複数回用
    # program = Iteration_multi(n,m,eta,pattern,count)
    # iteration_count = program.main()

    #一回グラフ作成用
    program = Iteration_multi_graph(n,m,eta,pattern,count)
    iteration_count = program.main()

    print(np.mean(iteration_count,axis=1))

print('finish2')