import numpy as np
from Regression.ronbun.ronbun_jikken2.iteration import Iteration_multi
from Regression.ronbun.ronbun_jikken2 import Solver

a,b,c,d,e = 'Harnessing','Proposed','Subgrad','ADMM','DA'
#####################################################################################
# 設定パラメータ
n = 25
m = 20
np.random.seed(0)  # ランダム値固定

count = 1

# algo = [a,b,c,d]]
algo = [d]
parameter = [(0.01,0.01)]
stop_condition = 0.01
#####################################################################################


#複数回用
program = Iteration_multi(n,m,parameter,algo,count,stop_condition)
iteration_count = program.main()

#一回グラフ作成用
# program = Iteration_multi_graph(n,m,parameter,pattern,algo,count)
# iteration_count = program.main()

print(np.mean(iteration_count,axis=1))

print('finish2')