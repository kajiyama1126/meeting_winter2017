import numpy as np
import matplotlib.pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import copy


class Diffusion(object):
    def __init__(self):
        self.h_x = 20
        self.h_y = 20

        self.x_div = 0.2
        self.y_div = 0.2

        self.x_size_div = int(self.h_x / self.x_div)
        self.y_size_div = int(self.h_y / self.y_div)

        self.time = 0

        self.map = None
    def make_map(self):
        self.map = np.zeros((self.x_size_div,self.y_size_div))
        self.u = np.zeros(self.x_size_div*self.y_size_div)


    def make_initial_distribution(self):
        for i in range(self.x_size_div):
            for j in range(self.y_size_div):
                x = self.x_div * i
                y = self.y_div * j
                self.map[j][i] = self.distribution_function(x,y)

    def distribution_function(self,x,y):
        tmp = math.exp(-0.1*((x-15)**2 + (y-10)**2))
        tmp1 = 0.5* math.exp(-0.2*((x-5)**2+(y-15)**2))
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
                    if i==0 or i == self.x_size_div-1 or j==0 or j == self.y_size_div-1:
                        self.map[j][i] =0
                    else:
                        self.map[j][i] = alpha * (map_bf[j+1][i]+map_bf[j-1][i]) + beta * (map_bf[j][i+1]+map_bf[j][i-1])+gamma * map_bf[j][i]



if __name__=='__main__':
    Map = Diffusion()
    Map.make_map()
    Map.make_initial_distribution()
    Map.draw()
    Map.update_1(1000)
    # Map.draw()
    Map.draw()