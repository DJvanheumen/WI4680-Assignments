import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plottingtools as st

from continuation import Fun_ext, run_continuation
from timeintegrator import time_integrator

mpl.use('TkAgg')
colours = st.configure()


# ============================================================================
# Functions
# ============================================================================
def integrate_trajectory(params, ic, dt, T, eps, Bmax):
    """RK4-integrate the extended (R, rho, B) system.

    params : (A, R0, rho0, deltaR, n)
    ic     : initial condition (R, rho, B)
    eps    : rate of change of B (eps = 0 keeps B fixed at ic[2])
    Returns t and y_tim of shape (3, len(t)) with rows [R, rho, B].
    """
    params_ext = (*params, Bmax, eps)
    return time_integrator(Fun_ext, params_ext, ic, dt, T)


def plot_trajectories(t, y_tim, B_arr, y_s, y_u, Bmax_bifdiag):
    """4-panel figure: equilibrium branch (stable/unstable) with the trajectory
    overlaid in the (B, R), (B, rho) and (R, rho) planes, plus B(t)."""
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)                      # Rac vs B
    plt.plot(B_arr, y_s[:, 0], '-', color=colours[0], label='stable')
    plt.plot(B_arr, y_u[:, 0], '--', color=colours[0], label='unstable')
    plt.plot(y_tim[2, :], y_tim[0, :], '-', color=colours[1], label='trajectory')
    plt.plot(y_tim[2, 0], y_tim[0, 0], 'o', color=colours[1])
    plt.xlabel('B'); plt.ylabel('Rac'); plt.ylim(0, 1); plt.xlim(0, Bmax_bifdiag)
    plt.legend(fontsize=7)

    plt.subplot(2, 2, 2)                      # rho vs B
    plt.plot(B_arr, y_s[:, 1], '-', color=colours[0])
    plt.plot(B_arr, y_u[:, 1], '--', color=colours[0])
    plt.plot(y_tim[2, :], y_tim[1, :], '-', color=colours[1])
    plt.plot(y_tim[2, 0], y_tim[1, 0], 'o', color=colours[1])
    plt.xlabel('B'); plt.ylabel('rho'); plt.ylim(0, 1); plt.xlim(0, Bmax_bifdiag)

    plt.subplot(2, 2, 3)                      # Rac vs rho
    plt.plot(y_s[:, 0], y_s[:, 1], '-', color=colours[0])
    plt.plot(y_u[:, 0], y_u[:, 1], '--', color=colours[0])
    plt.plot(y_tim[0, :], y_tim[1, :], '-', color=colours[1])
    plt.plot(y_tim[0, 0], y_tim[1, 0], 'o', color=colours[1])
    plt.xlabel('Rac'); plt.ylabel('rho'); plt.ylim(0, 1); plt.xlim(0, 1)

    plt.subplot(2, 2, 4)                      # B vs t
    plt.plot(t, y_tim[2, :], '-', color=colours[1])
    plt.xlabel('t'); plt.ylabel('B')

    plt.tight_layout()


# ============================================================================
# Main body
# ============================================================================
if __name__ == '__main__':
    # --- parameters ---
    params = (0.003, 0.3, 0.16, 1, 4)               # A, R0, rho0, deltaR, n
    B0_cont, Bmax_bifdiag, ds = 0.001, 0.1, 0.001   # continuation start, end, step
    ic = (0.1, 0.05, 0.03)                          # initial condition (R, rho, B)
    dt, T = 0.01, 100                               # time step, horizon
    eps = 0.0                                       # unperturbed: B fixed at ic[2]

    # --- results ---
    B_arr, y_arr, y_s, y_u, evs, all_evs = run_continuation(params, B0_cont, Bmax_bifdiag, ds)
    t, y_tim = integrate_trajectory(params, ic, dt, T, eps,Bmax_bifdiag)

    # --- plot / print ---
    print(f"endpoint: R={y_tim[0, -1]:.4f}, rho={y_tim[1, -1]:.4f}")
    plot_trajectories(t, y_tim, B_arr, y_s, y_u, Bmax_bifdiag)
    st.show()
