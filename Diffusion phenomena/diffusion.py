import numpy as np
import matplotlib.pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import copy


class Diffusion(object):
    def __init__(self):
        self.h_x = 5
        self.h_y = 4

        self.x_div = 1
        self.y_div = 1

        self.x_size_div = int(self.h_x / self.x_div)
        self.y_size_div = int(self.h_y / self.y_div)

        self.time = 0

        self.map = None

    def make_Omega(self):
        alpha = 0.1
        beta = 0.1
        gamma = 1-2*alpha - 2*beta

        h = int((self.x_size_div)*(self.y_size_div))



        # Im_y = np.identity(self.h_y-1)
        # Theta = gamma * np.identity(self.h_y-1)
        # for i in range(self.h_y-1):
        #     for j in range(self.h_y - 1):
        #         Theta[i][j] = beta
        # Om_y = np.zeros(self.h_y-1)



        # self.Omega = np.zeros((self.h_x-1,self.h_x-1))
        self.Omega = np.zeros((h,h))
        # print(self.Omega)
        for i in range(h):
            for j in range(h):
                if i== j:
                    self.Omega[i][j] = gamma
                elif i == j+1:
                    self.Omega[i][j] = beta
                elif i == j-1:
                    self.Omega[i][j] = beta
                elif j == (self.y_size_div)+i:
                    self.Omega[i][j] = alpha
                elif j == -(self.y_size_div)+i:
                    self.Omega[i][j] = alpha

        # print(self.Omega)
        # p = [i for i in range(self.h_x-1)]
        # q = [i for i in range(self.h_y-1)]

        # for i in range(h):
        #     for j in range(h):


    def make_map(self):
        self.map = np.zeros((self.y_size_div,self.x_size_div))
        self.u = np.zeros(self.x_size_div*self.y_size_div)


    def make_initial_distribution(self):
        for i in range(self.x_size_div):
            for j in range(self.y_size_div):
                x = self.x_div * i
                y = self.y_div * j
                tmp = self.distribution_function(x,y)
                self.map[j][i] = tmp
                self.u[self.y_size_div*i+j]  = tmp

    def distribution_function(self,x,y):
        tmp = 10*math.exp(-1*((x-5)**2 + (y-4)**2))
        # tmp1 = 0.5* math.exp(-0.2*((x-5)**2+(y-3)**2))
        tmp1 = 0
        return tmp + tmp1

    def draw(self):
        x = np.arange(0, self.h_x, self.x_div)
        y= np.arange(0, self.h_y, self.y_div)
        X,Y = plt.meshgrid(x,y)
        Z = self.map

        plt.pcolor(X,Y,Z)
        plt.colorbar()
        plt.show()

    def draw3D(self):
        x = np.arange(0, self.h_x, self.x_div)
        y= np.arange(0, self.h_y, self.y_div)
        X,Y = plt.meshgrid(x,y)
        Z = self.map

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.plot_wireframe(X,Y,Z)

        plt.show()


    # def update(self):

        # h_x = int(self.x_size/self.x_div)
        # h_y = int(self.y_size/self.y_div)
        # h = (h_x-1)*(h_y-1)
        #
        # self.Omega = np.zeros((self.h,self.h))
        # for i in range(self.h):
        #     for j in range(self.h):
        #         if i+1 == j or i-1 == j:
        #             self.Omega[i][j]


    def update_1(self,iteration):
        # self.map_update = copy.copy(self.map)
        alpha = 0.2
        beta = 0.2
        gamma = 1-2*alpha - 2*beta

        for k in range(iteration):
            map_bf = copy.copy(self.map)
            for i in range(self.x_size_div):
                for j in range(self.y_size_div):
                    if i==0 or j==0:
                        self.map[j][i] =alpha * (map_bf[j+1][i]) + beta * (map_bf[j][i+1])+gamma * map_bf[j][i]
                    else:
                        self.map[j][i] = alpha * (map_bf[j+1][i]+map_bf[j-1][i]) + beta * (map_bf[j][i+1]+map_bf[j][i-1])+gamma * map_bf[j][i]

    def update_2(self,iteration):
        self.make_Omega()
        for k in range(iteration):
            self.u = np.dot(self.Omega,self.u)
            # print(self.u)
        self.map = np.reshape(self.u,(self.y_size_div,self.x_size_div),order='F')


#今後考える
if __name__=='__main__':
    Map = Diffusion()
    Map.make_map()
    Map.make_initial_distribution()
    # Map.make_Omega()
    Map.update_1(10)
    up1 = Map.map
    Map.draw()
    Map2 = Diffusion()
    Map2.make_map()
    Map2.make_initial_distribution()
    Map.make_Omega()
    Map2.update_2(10)
    up2 = Map2.map

    if np.allclose(up1,up2,rtol=0.01):
        print('success')
    else:
        print('error')
        print(up1,up2)
    # Map.draw3D()
    # Map.draw()
