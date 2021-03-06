import matplotlib.pyplot as plt
import cvxpy as cvx
from progressbar import ProgressBar

from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken3.condition_check import *
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken3.make_communication import Circle_communication
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken3.ronbun_jikken3_agent import Agent_YiHong14
from quanzize.TAC2019_2019_0620.Regression.ronbun.ronbun_jikken3.ronbun_jikken3_agent import \
    Agent_harnessing_quantize_add_send_data_ronbun_jikken3 as Agent_jikken3

np.random.seed(0)
#Parameter_setting = True
Parameter_setting = False

# 共通parameter============================================================
patterns = 2
#iteration = 30000
#iteration = 120000
iteration = 300000
n = 4
m = 2
K = 1  # 先行研究でのellに相当(量子化レベル)
#####################################################################################
# f_i = ||Ax-b||_2^2の場合
A1 = np.array([[0.7, 0.2], [0.3, 0.6]])
# A1 = np.array([[1, 2], [1, 1]])
#b = np.array([-1.5, 0.5]) + 1.0 * np.random.rand(2)

alpha = np.linalg.norm(np.dot(A1.T, A1),2)
beta = np.linalg.norm(np.dot(A1.T, A1),2)
print(alpha, beta)
C_g = np.linalg.norm(np.dot(np.dot(A1.T, A1),np.array([-1,1])),2)
#####################################################################################

# ==========================================================================
#3段階
# 提案手法parameter===========================================================
#mu = 0.99992
# mu = 0.9999
# C_x = 0.45
# C_v = 0.90

#w = 0.0008
#eta = 0.000388

#w = 0.001
#eta = 0.000555

# mu = 0.999935
# C_x = 0.5
# C_v = 10/9
# w = 0.0005
# eta = 0.0002

mu = 0.99992
C_x = 0.45
C_v = 0.9

w = 0.0008
eta = 0.00038

delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
delta_x = 0

C_x0 = 0  # max_i ||x_i(0)||
C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||

# =========================================================================
# #63段階
# # 提案手法parameter===========================================================
# mu = 0.9968
# C_x = 2.1
# C_v = 4.1
#
# w = 0.021
# eta = 0.0088
#
# delta_ast = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A1.T,A1)),A1.T),2)*(1. ** 2 + 1. ** 2) ** 0.5 * n ** 0.5
# delta_v = np.linalg.norm(A1,2)*(1**2+1**2)**0.5
# delta_x = 0
#
# C_x0 = 0  # max_i ||x_i(0)||
# C_v0 = np.linalg.norm(A1,2)*(1.**2 + 1.**2)**0.5  # max_i ||v_i(0)||
# # =========================================================================
# 先行研究(劣勾配)============================================================
w_2 = 0.25
s_0 = 10
gamma = 0.99

C_x2 = 0  # max_i ||x_i(0)||
C_delta2 = 0  # max_i,j ||x_i(0)-x_j(0)||
# ============================================================================


Graph = Circle_communication(n, w)
Graph.make_circle_graph()
Weight_matrix = Graph.send_P()

Graph_2 = Circle_communication(n, w_2)
Graph_2.make_circle_graph()
Laplacian_matrix = Graph_2.send_L()

Condition_proposed(n, m, Weight_matrix, w, eta, C_x, C_v, mu, alpha, beta, delta_x, delta_v, delta_ast, C_x0, C_v0)
print('Proposed_settings')
Condition_prior(n, m, Laplacian_matrix, w_2, s_0, gamma, C_x2, C_delta2, C_g, K)

if Parameter_setting is True:
    sys.exit()

A = []
B = []

for i in range(n):
    A.append(A1)
    b = np.array([-1, 0]) + 1.0 * np.random.rand(2)
    # if i == 0 or i ==1:
    #     b = np.array([-1,1,1])
    # elif i == 2:
    #     b =  np.array([1,0,-1])
    # elif i ==3:
    #     b = np.array([3, 1, -3])
    B.append(b)

def solve(A,B,m):

    x = cvx.Variable(m)
    obj = cvx.Minimize(0)
    for i in range(len(A)):
        obj += cvx.Minimize(1/2*cvx.power(cvx.norm(A[i]*x-B[i],2),2))
    pro = cvx.Problem(obj)
    pro.solve(abstol=1.0e-10, feastol=1.0e-10)
    return x.value, pro.value

x_opt,f_opt = solve(A,B,m)
print(x_opt,f_opt)



ydata_set = []
zdata_set = []

for pattern in range(patterns):
    Agents = []
    sumf_list = []
    prog = ProgressBar(max_value=iteration)
    for i in range(n):
        if pattern < int(patterns) / 2:
            Agents.append(Agent_jikken3(n, m, A[i], B[i], eta, Weight_matrix[i], i, C_x, C_v, mu))
        else:
            Agents.append(Agent_YiHong14(n, m, A[i], B[i], Laplacian_matrix[i],w_2, i))

    for k in range(iteration):
        prog.update(k + 1)

        for i in range(n):
            for j in range(n):
                x_j, name = Agents[i].send(j)
                Agents[j].receive(x_j, name)

        for i in range(n):
            Agents[i].update(k)

        f_value = []

        for i in range(n):
            f_sum = 0
            for j in range(n):
                f_sum += 1 / 2 * np.linalg.norm(np.dot(A[j],Agents[i].x_i) - B[j], 2) ** 2
            f_value.append(f_sum)

        sumf_list.append(max(f_value) - f_opt)
        # print(max(f_value)-f_opt)
        if k % 10000 == 0:
            print('iteration', k)
            for i in range(n):
                print(Agents[i].x_i)

    for i in range(n):
        print(Agents[i].x_i)

    dim_label = ['Proposed', '[35]']
    dim_color = ['g','r']
    plt.plot(sumf_list, label=dim_label[pattern],color=dim_color[pattern])
    plt.yscale('log')

    plt.grid(which='major', color='gray', linestyle=':')
    plt.grid(which='minor', color='gray', linestyle=':', axis='y')
    #plt.grid(which="both", color='gray', linestyle=':')
    plt.minorticks_on()
    plt.xlabel('iteration $k$', fontsize=12)
    plt.ylabel('$max_{i} f(x_i(k))-f^*$', fontsize=12)
    y_data, z_data = Agents[0].send_y_data_zdata()
    ydata_set.append(y_data)
    zdata_set.append(z_data)

plt.tick_params(labelsize=10)
plt.legend(fontsize=10)
plt.xlim([0, iteration])
#plt.xlim([0, 30000])
#plt.ylim([0.000001, 10])
plt.ylim([0.00000001, 10])
#plt.ylim([10**(-10), 10**(-1)])
plt.tight_layout()
plt.savefig("cost_3bit.png")
plt.savefig("cost_3bit.eps")
plt.legend()
plt.show()

for pattern in range(patterns):
    y_data = ydata_set[pattern]
    z_data = zdata_set[pattern]

    for i in range(1):
        if y_data[i][0] == []:
            pass
        else:
            dim_label = ['1', '2']
            for j in range(m):
                plt.plot(y_data[i][j], 'o', markersize=4,color=dim_color[pattern])
                plt.xlabel('iteration $k$', fontsize=18)
                plt.ylabel('$y_{12}^' + dim_label[j] + '$', fontsize=18)
                # plt.legend()
                plt.show()
                # plt.legend()
                # plt.show()
                if z_data is not None:
                    plt.plot(z_data[i][j], 'o', markersize=4,color=dim_color[pattern])
                    plt.xlabel('iteration $k$', fontsize=18)
                    plt.ylabel('$z_{12}^' + dim_label[j] + '$', fontsize=18)
                    # plt.legend()
                    plt.show()
#             break
