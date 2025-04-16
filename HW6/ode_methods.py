import numpy as np

# Euler's method of solving 1st order ODE

def euler(f, a, b, n):

    h  = (b - a)/n
    x  = np.arange(a, b + h, h)
    y  = np.zeros(n + 1)

    x[0] = a
    y[0] = 2

    for i in range(0, n):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h

    np.savetxt('euler.txt', np.transpose([x, y]), fmt = '%10.4f')


# Modified Euler's method of solving 1st order ODE

def euler_mod(f, a, b, n):

    h    = (b - a)/n
    x    = np.arange(a, b + h, h)
    y    = np.zeros(n + 1)
    yEu  = np.zeros(n + 1)

    x[0] = a   
    y[0] = 2

    for i in range(0, n):
        x[i+1]   = x[i] + h
        yEu[i+1] = y[i] + h * f(x[i], y[i])
        y[i+1]   = y[i] + ( f(x[i], y[i]) + f(x[i+1], yEu[i+1]) ) * h/2

    np.savetxt('euler_mod.txt', np.transpose([x, y]), fmt = '%10.4f')


# Runge-Kutta 4th order method of solving 1st order ODE

def rk4(f, a, b, n):

    h    = (b - a)/n
    x    = np.arange(a, b + h, h)
    y    = np.zeros(n + 1)

    x[0] = a
    y[0] = 2

    for i in range(0, n):
        k1 = f( x[i], y[i] )
        k2 = f( x[i] + h/2, y[i] + k1*h/2 )
        k3 = f( x[i] + h/2, y[i] + k2*h/2 )
        k4 = f( x[i] + h,   y[i] + k3*h   )

        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)*h/6

    np.savetxt('rk4.txt', np.transpose([x, y]), fmt = '%10.4f')







