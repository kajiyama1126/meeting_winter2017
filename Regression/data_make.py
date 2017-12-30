import matplotlib.pyplot as plt
import numpy as np

# 回帰曲線の練習

def make_ydata(x):
    # return 3*(x**3 -4/3*x**2 +1/3*x)+1
    return x ** 2 - 2*x - 1


x = np.array([0.01 * i for i in range(300)])
sensor_x = np.array([0.1 * i for i in range(30)])
y = make_ydata(x)
sensor_y1 = make_ydata(sensor_x) + 0.1 * np.random.randn(len(sensor_x)) + 0.03*sensor_x
sensor_y2 = make_ydata(sensor_x) + 0.1 * np.random.randn(len(sensor_x)) - 0.1 * sensor_x
sensor_y3 = make_ydata(sensor_x) + 0.1 * np.random.randn(len(sensor_x)) + 0.07*sensor_x
# sensor_y1 = make_ydata(sensor_x)
# sensor_y2 = make_ydata(sensor_x)
# sensor_y3 = make_ydata(sensor_x)
n = 3
x_i = np.reshape(np.zeros(n), (-1, 1))

sensor_y = [sensor_y1, sensor_y2, sensor_y3]
print(x_i)


def make_phi(n, x_data):
    phi = np.array([[(i ** (j)) for j in range(n)] for i in x_data])
    return phi


phi = make_phi(n, sensor_x)


def grad(phi, x_i, y):
    phi_to = phi.T
    grad = np.dot(phi_to, np.dot(phi, x_i) - y)
    return grad


graph = [[0.5, 0.25, 0.25],
         [0.25, 0.5, 0, 25],
         [0.25, 0.25, 0.5]]

class Agent(object):
    def __init__(self,n ,weight,y,phi):
        self.y = np.reshape(y,(-1,1))
        self.weight = weight
        self.phi =phi
        self.x_i = np.reshape(np.zeros(n),(-1,1))
        self.eta = 0.0005


    def grad(self):
        phi_to = self.phi.T
        grad = np.dot(phi_to, np.dot(self.phi, self.x_i) -self.y)
        return grad

    def update(self):
        self.x_i = self.x_i -self.eta * self.grad()

agents = []

for i in range(3):
    agents.append(Agent(n,graph[i],sensor_y[i],phi))

for k in range(10000):
    for i in range(3):
        agents[i].update()
        print(agents[i].x_i)

def kaiki_graph(n_dim,x_n,weight):
    matrix = np.array([[i ** j for j in range(n_dim)] for i in x_n])
    return np.dot(matrix,weight)
x_n = [0.01 * i for i in range(300)]

for i in range(3):
    kaiki_line = kaiki_graph(n,x_n,agents[i].x_i)
    plt.plot(x_n,kaiki_line)


# # def update(x_i):
# sensor_y_data = np.reshape(sensor_y,(-1,1))
# for i in range(10):
#     x_i += 0.0000002 * (-grad(phi,x_i,sensor_y_data))
#     print(x_i)



plt.plot(x, y)
plt.scatter(sensor_x, sensor_y1)
plt.scatter(sensor_x, sensor_y2)
plt.scatter(sensor_x, sensor_y3)
plt.show()
