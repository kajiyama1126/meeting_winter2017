import numpy as np
from Regression.ronbun.ronbun_jikken2.iteration import Iteration_multi_graph,Iteration_multi
#from Regression.ronbun.ronbun_jikken2.iteration import Iteration_multi
from Regression.ronbun.ronbun_jikken2 import Solver

a,b,c,d,e = 'Proposed','Harnessing','DA','Subgrad','ADMM'
#####################################################################################
# 設定パラメータ
n = 25
m = 20
np.random.seed(0)  # ランダム値固定

count = 1

algo = [a,c,d,e] #<--
#parameter = [0.01, 10, None, (0.05, 0.001)] #<--
#parameter = [0.01, 10, None, (0.0001, 0.001)] #<--
parameter = [0.01, 20, None, (0.00001, 0.0001)] #<--
#parameter = [0.01, 10, None, (0.1, 0.1)] #<--

#algo = [c]
#parameter = [10]

#algo = [d]
#parameter = [0.01]

#algo = [e]
#parameter = [(0.06, 0.0001)]

####---------------------####
#algo = [a,b,c,d]
#algo = [a,b,c,d]
#algo = [a,b,c,d,e]

#parameter = [0.01, 0.01, None, (0.001, 0.02)]
#parameter = [0.01, 0.01, None, (0.001, 0.001)]
#parameter = [0.01, 0.01, None, (0.01, 0.001)] #<--
#parameter = [0.01, 0.01, None, (0.01, 0.001), 10] #<--
#parameter = [0.01, 0.01, 10, None, (0.01, 0.001)] #<--
#parameter = [2]
#stop_condition = 0.01
#pattern = len(algo)
#####################################################################################


#複数回用
#program = Iteration_multi(n,m,parameter,algo,count,stop_condition)
#iteration_count = program.main()

#一回グラフ作成用#
#program = Iteration_multi_graph(n,m,parameter,algo,count)
#program = Iteration_multi_graph(n,m,parameter,algo,count,stop_condition)
program = Iteration_multi_graph(n,m,parameter,algo,count)
#program = Iteration_multi(n,m,parameter,algo,count,stop_condition)
# program = Iteration_multi_graph(n,m,parameter,algo,count)
iteration_count = program.main()

print(np.mean(iteration_count,axis=1))

print('finish2')