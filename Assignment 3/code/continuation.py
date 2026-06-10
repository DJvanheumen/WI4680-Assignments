"""
Model code for the Rac-Rho model
Y.M. Dijkstra, 2024
"""
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import plottingtools as st
from timeintegrator import time_integrator
from rootfinding import newton, newton_small
import numpy as np
colours = st.configure()
from copy import copy

"""
########################################################################################################################
Functions and Jacobians
########################################################################################################################
"""
def fun(y, B, params):              ## Functions for Rac and Rho, returns the d/dt
    A, R0, rho0, deltaR, n = params
    R = y[0]
    rho = y[1]
    F = np.zeros(2)
    F[0] = A/(rho0**n + rho**n)*(1-R)-deltaR*R
    F[1] = B/(R0**n + R**n)*(1-rho)-rho
    return F

def fun_ext(y, params):         ## Extended function with parameter; dB/dt = eps  (Q3: linear, unbounded drift)
    A, R0, rho0, deltaR, n, Bmax, eps = params
    F = np.zeros(3)
    B = y[2]
    F[:2] = fun(y[:2], B, (A, R0, rho0, deltaR, n))
    ##### adapt the code here for changing dB/dt
    F[2] = eps
    #####
    return F

def fun_ext_sat(y, params):     ## Extended function; dB/dt = eps*(Bmax-B)/Bmax  (Q4: B saturates at Bmax -> rate-induced tipping)
    A, R0, rho0, deltaR, n, Bmax, eps = params
    F = np.zeros(3)
    B = y[2]
    F[:2] = fun(y[:2], B, (A, R0, rho0, deltaR, n))
    F[2] = eps*(Bmax - B)/Bmax
    return F

def jac(y, B, params):          ## Jacobian
    A, R0, rho0, deltaR, n = params
    R = y[0]
    rho = y[1]
    J = np.zeros((2,2))
    J[0,0] = -A/(rho0**n+rho**n)-deltaR
    J[0,1] = -A*(1-R)*n*rho**(n-1)/(rho0**n+rho**n)**2
    J[1,0] = -B*(1-rho)*n*R**(n-1)/(R0**n+R**n)**2
    J[1,1] = -B/(R0**n+R**n)-1
    return J

def jac_B(y, B, params):        ## Derivative w.r.t. B
    A, R0, rho0, deltaR, n = params
    R = y[0]
    rho = y[1]
    JB = np.zeros(2)
    JB[1] = (1-rho)/(R0**n+R**n)
    return JB

"""
########################################################################################################################
Continuation routine (pseudo-arclength) -- importable so main.py can drive it
########################################################################################################################
"""
def run_continuation(params, B0, Bmax_bifdiag, ds):
    """Trace the equilibrium branch of the (R, rho) system as B varies.

    Parameters (all set by the caller, e.g. in main.py)
    ----------
    params        : tuple (A, R0, rho0, deltaR, n)
    B0            : starting value of B for the continuation
    Bmax_bifdiag  : maximum value of B; the continuation terminates beyond this
    ds            : arclength step size

    Returns
    -------
    B_arr   : array of B values along the branch
    y_arr   : (len(B), 2) equilibria, columns [Rac, rho]
    y_s     : y_arr with unstable points set to nan (stable branch)
    y_u     : y_arr with stable points set to nan (unstable branch)
    evs     : max real part of the Jacobian eigenvalues along the branch
    all_evs : (len(B), 2) real parts of both eigenvalues along the branch
    """
    B_arr = []      # parameter B
    y_arr = []      # solution y
    evs = []        # largest eigenvalues
    all_evs = []    # list of all eigenvalues

    # Find equilibrium for B = B0 (newton on the original 2D system)
    y0 = np.array([1, 1])
    B = B0
    y = newton_small(fun, jac, y0, B, params)

    # add results for B0 to the array
    B_arr.append(B)
    y_arr.append(y)
    ev = np.linalg.eigvals(jac(y, B, params))
    evs.append(np.max(np.real(ev)))
    all_evs.append((np.real(ev)))

    # start continuation
    iter = 0
    while B < Bmax_bifdiag and iter < 3000:
        # predictor
        J = np.zeros((3, 3))
        rhs = np.zeros(3)
        J[:2, :2] = jac(y, B, params)
        J[:2, -1] = jac_B(y, B, params)
        J[-1, 1] = 1
        rhs[-1] = 1
        z = np.linalg.solve(J, rhs)             # z contains components da/ds and dmu/ds
        z_rescale = z / np.sqrt(np.sum(z**2))   # scaled s.t. total step length is ds
        y_pred = y + z_rescale[:-1] * ds
        B_pred = B + z_rescale[-1] * ds

        # corrector
        y, B = newton(fun, jac, jac_B, y_pred, B_pred, y, B, ds, params)

        # eigenvalues for assessing stability
        ev = np.linalg.eigvals(jac(y, B, params))
        evs.append(np.max(np.real(ev)))         # save maximum real value of evs
        all_evs.append((np.real(ev)))

        # update output arrays
        B_arr.append(B)
        y_arr.append(y)
        iter += 1

    # convert lists to arrays
    B_arr = np.asarray(B_arr)
    y_arr = np.asarray(y_arr)       # shape (len(B), 2), 2nd dim = [Rac, rho]
    evs = np.asarray(evs)
    all_evs = np.asarray(all_evs)

    # split branch into stable ('y_s') and unstable ('y_u')
    y_s = copy(y_arr)
    y_s[np.where(evs > 0)[0], :] = np.nan
    y_u = copy(y_arr)
    y_u[np.where(evs < 0)[0], :] = np.nan

    return B_arr, y_arr, y_s, y_u, evs, all_evs


"""
########################################################################################################################
Reference run: reproduces the original Figure 1 when this file is executed directly
########################################################################################################################
"""
if __name__ == '__main__':
    # Default dimensionless parameters
    A = 0.003
    R0 = 0.3
    rho0 = 0.16
    deltaR = 1
    n = 4
    ds = 0.001
    B0 = 0.001
    Bmax_bifdiag = 0.1      # maximum value of B for continuation; terminate for larger B

    eps = 0.0               # rate of change of B, used in exercise 3, 4
    Bmax = 0.04             # value Bmax, only used in exercise 4

    dt = 0.01
    T = 10

    params = (A, R0, rho0, deltaR, n)
    params_ext = (A, R0, rho0, deltaR, n, Bmax, eps)

    # continuation
    B_arr, y_arr, y_s, y_u, evs, all_evs = run_continuation(params, B0, Bmax_bifdiag, ds)

    # the time integrator has arguments: function, parameter list, tuple of initial conditions (R, rho, B), time step, end time.
    # y_tim has shape (3, length(t)). In the first dimension it has entries Rac, rho, B
    t, y_tim = time_integrator(fun_ext, params_ext, (0.1, 0.05, 0.03), dt, T)

    ########################################################################################################################
    # Plot
    ########################################################################################################################
    ## Figure 1 ##
    plt.figure(1, figsize=(2,2))
    plt.subplot(2,2,1)
    # Plot result of continuation Rac vs B
    plt.plot(B_arr, y_s[:,0], '-', color=colours[0])
    plt.plot(B_arr, y_u[:,0], '--', color=colours[0])

    # Plot trajectory in same plot
    plt.plot(y_tim[-1,:], y_tim[0,:], '-', color=colours[1])    # trajectory y_tim[-1,:] = B(t), y_tim[0,:] = Rac(t)
    plt.plot(y_tim[-1,0], y_tim[0,0], 'o', color=colours[1])    # mark initial condition
    plt.xlabel('B')
    plt.ylabel('Rac')
    plt.ylim(0, 1)
    plt.xlim(0, Bmax_bifdiag)

    plt.subplot(2,2,2)
    # Plot result of continuation rho vs B
    plt.plot(B_arr, y_s[:,1], '-', color=colours[0])
    plt.plot(B_arr, y_u[:,1], '--', color=colours[0])

    # Plot trajectory in same plot
    plt.plot(y_tim[-1,:], y_tim[1,:], '-', color=colours[1])    # trajectory y_tim[-1,:] = B(t), y_tim[0,:] = rho(t)
    plt.plot(y_tim[-1,0], y_tim[1,0], 'o', color=colours[1])    # mark initial condition
    plt.xlabel('B')
    plt.ylabel('rho')
    plt.ylim(0, 1)
    plt.xlim(0, Bmax_bifdiag)

    plt.subplot(2,2,3)
    # Plot result of continuation Rac vs rho
    plt.plot(y_s[:,0], y_s[:,1], '-', color=colours[0])
    plt.plot(y_u[:,0], y_u[:,1], '--', color=colours[0])

    # Plot trajectory in same plot
    plt.plot(y_tim[0,:], y_tim[1,:], '-', color=colours[1])
    plt.plot(y_tim[0,0], y_tim[1,0], 'o', color=colours[1])
    Bplot = 1
    plt.xlabel('Rac')
    plt.ylabel('rho')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.subplot(2,2,4)
    # B as function of t
    plt.plot(t, y_tim[2,:], '-', color=colours[1])
    plt.ylabel('B')
    plt.xlabel('t')

    st.show()