"""
Time integrator using Runge-Kutta 4
"""
import numpy as np


def time_integrator(modelfun, params, x0, dt, T):
    t = np.arange(0, T, dt)
    x = np.zeros((len(x0), len(t)))
    x[:, 0] = x0

    for i in range(1,len(t)):

        ## RK4
        k1 = dt*modelfun(x[:,i-1],params)
        k2 = dt*modelfun(x[:,i-1]+k1/2,params)
        k3 = dt*modelfun(x[:,i-1]+k2/2,params)
        k4 = dt*modelfun(x[:,i-1]+k3,params)

        dx = 1/6*(k1+2*k2+2*k3+k4)
        x[:, i] = (x[:, i-1]+dx)

    return t, x