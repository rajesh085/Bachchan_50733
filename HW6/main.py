import numpy as np
import matplotlib.pyplot as plt
from ode_methods import euler, euler_mod, rk4

f = lambda x, y: -y + 5*np.exp(-0.4*x)

a = 0.0    # start point
b = 2.5    # end point
n = 50     # number of points

euler(f, a, b, n)           # use the Euler method
euler_mod(f, a, b, n)       # use the Euler-modified method
rk4(f, a, b, n)             # use the Runge-Kutta 4th order method

# comparing the three methods
data1  = np.loadtxt('euler.txt')
data2  = np.loadtxt('euler_mod.txt')
data3  = np.loadtxt('rk4.txt')

plt.plot(data1[:, 0], data1[:, 1]) 
plt.plot(data2[:, 0], data2[:, 1], 'o') 
plt.plot(data3[:, 0], data3[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Euler', 'Euler modified', 'Runge-Kutta'])
plt.grid()
plt.savefig('ODE_comparison.pdf')





























