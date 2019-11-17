# -*- coding: utf-8 -*-
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np


class Communication:  # (頂点数，辺数，辺確率)
    def __init__(self, n, k, p):
        self.n = n
        self.k = k
        self.p = p
        self.count = 0

    def make_connected_WS_graph(self, count):
        self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        self.weight_martix()

        # Adj = np.loadtxt('Adj_wb.txt', delimiter=",")
        # self.G = nx.from_numpy_matrix(Adj, create_using=nx.DiGraph())
        # self.G = nx.DiGraph(self.G)
        # pos = nx.spring_layout(self.G)
        # nx.draw_networkx_nodes(self.G, pos, node_size=30, alpha=1.0, node_color="blue")
        # nx.draw_networkx_edges(self.G, pos, width=0.6, arrowstyle='->', arrowsize=10)
        # plt.axis('off')
        # plt.savefig("network{0}.png".format(count))
        # plt.savefig("network{0}.eps".format(count))
        # plt.show()
        # self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        # self.weight_martix()

    def weight_martix(self):
        a = np.zeros(self.n)
        for i in range(self.n):
            a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))

        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A[i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum

    def send_P(self):
        return self.P


class Circle_communication(object):
    def __init__(self,n,w):
        self.n = n
        self.w = w
    def make_circle_graph(self):
        self.G = nx.cycle_graph(self.n)
        self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
        self.weight_martix()

    def weight_martix(self):
        a = np.zeros(self.n)
        for i in range(self.n):
            # a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
            a[i]  = self.w
        self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j and self.A[i][j] == 1:
                    a_ij = min(a[i], a[j])
                    self.P[i][j] = copy.copy(a_ij)
                    self.P[j][i] = copy.copy(a_ij)


                #         print(self.P)
        for i in range(self.n):
            sum = 0.0
            for j in range(self.n):
                sum += self.P[i][j]
            self.P[i][i] = 1.0 - sum

    def send_P(self):
        return self.P

if __name__ == '__main__':
    graph = Communication(10,2,0.3)
    graph.make_connected_WS_graph()
    print(graph.P)



# # -*- coding: utf-8 -*-
# import copy
#
# import networkx as nx
# import numpy as np
#
#
# class Communication:  # (頂点数，辺数，辺確率)
#     def __init__(self, n, k, p):
#         self.n = n
#         self.k = k
#         self.p = p
#         self.count = 0
#
#     def make_connected_WS_graph(self):
#         self.G = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
#         #         lam = nx.laplacian_spectrum(G)
#         #         print(nx.adjacency_matrix(G))
#         #         print (number_of_nodes(G))
#         #         (nx.degree(G))
#         # print(self.G)
#         self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
#         self.weight_martix()
#
#     #         print(self.A)
#
#     # def make_graph(self,number):
#     #     graph = [nx.dense_gnm_random_graph(self.n,self.m) for i in range(number)]
#
#     def weight_martix(self):
#         a = np.zeros(self.n)
#         for i in range(self.n):
#             a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
#
#         self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
#         for i in range(self.n):
#             for j in range(i, self.n):
#                 if i != j and self.A[i][j] == 1:
#                     a_ij = min(a[i], a[j])
#                     self.P[i][j] = copy.copy(a_ij)
#                     self.P[j][i] = copy.copy(a_ij)
#
#
#                 #         print(self.P)
#         for i in range(self.n):
#             sum = 0.0
#             for j in range(self.n):
#                 sum += self.P[i][j]
#             self.P[i][i] = 1.0 - sum
#
#     def send_P(self):
#         return self.P
#
#
# class Circle_communication(object):
#     def __init__(self,n,w):
#         self.n = n
#         self.w = w
#     def make_circle_graph(self):
#         self.G = nx.cycle_graph(self.n)
#         self.A = np.array(nx.adjacency_matrix(self.G).todense())  # 隣接行列
#         self.weight_martix()
#
#     def weight_martix(self):
#         a = np.zeros(self.n)
#         for i in range(self.n):
#             # a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
#             a[i]  = self.w
#         self.P = np.zeros((self.n, self.n))  # 確率行列(重み付き)
#         for i in range(self.n):
#             for j in range(i, self.n):
#                 if i != j and self.A[i][j] == 1:
#                     a_ij = min(a[i], a[j])
#                     self.P[i][j] = copy.copy(a_ij)
#                     self.P[j][i] = copy.copy(a_ij)
#
#
#                 #         print(self.P)
#         for i in range(self.n):
#             sum = 0.0
#             for j in range(self.n):
#                 sum += self.P[i][j]
#             self.P[i][i] = 1.0 - sum
#
#     def send_P(self):
#         return self.P
#
# if __name__ == '__main__':
#     graph = Communication(10,2,0.3)
#     graph.make_connected_WS_graph()
#     print(graph.P)