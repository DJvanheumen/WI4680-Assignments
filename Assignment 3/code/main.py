import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plottingtools as st

from continuation import fun_ext, run_continuation
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
    return time_integrator(fun_ext, params_ext, ic, dt, T)


def turning_point_indices(v, tol_frac=0.02):
    """Indices where the trajectory component v reverses direction, i.e. an
    overshoot/undershoot beyond both endpoints. Returns the global extremum
    index when it sticks out by more than tol_frac of the range, so it ignores
    numerical jitter near the equilibrium and monotonic components."""
    rng = np.nanmax(v) - np.nanmin(v)
    if rng == 0:
        return []
    tol = tol_frac * rng
    idx = []
    if np.nanmax(v) > max(v[0], v[-1]) + tol:
        idx.append(int(np.nanargmax(v)))
    if np.nanmin(v) < min(v[0], v[-1]) - tol:
        idx.append(int(np.nanargmin(v)))
    return idx


def _draw_arrow(x, y, s, frac, n_heads, dfrac=0.03):
    """Draw n_heads chevrons in succession from arc-length fraction frac,
    pointing in the direction of increasing time. A double chevron marks fast
    flow, a single chevron slow flow. s is the cumulative arc length."""
    for j in range(n_heads):
        i1, i2 = np.searchsorted(s, [(frac + j * dfrac) * s[-1],
                                     (frac + (j + 1) * dfrac) * s[-1]])
        i2 = max(i2, i1 + 1)
        plt.annotate('', xy=(x[i2], y[i2]), xytext=(x[i1], y[i1]),
                     arrowprops=dict(arrowstyle='-|>', color=colours[1],
                                     lw=1, mutation_scale=15))


def draw_bounce(xp, yp):
    """Mark an overshoot peak with a 'ceiling + downward arrow' (mapsto-like):
    a short horizontal bar the path rises to, with an arrow showing it then
    turns and descends. Sized relative to the current axis limits."""
    ax = plt.gca()
    hw = 0.045 * (ax.get_xlim()[1] - ax.get_xlim()[0])   # half-width of ceiling
    drop = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # short stub under the ceiling
    plt.plot([xp - hw, xp + hw], [yp, yp], '-', color=colours[1], lw=1.5)
    plt.annotate('', xy=(xp, yp - drop), xytext=(xp, yp),
                 arrowprops=dict(arrowstyle='-|>', color=colours[1],
                                 lw=1, mutation_scale=15))


def draw_trajectory(x, y, label=None, arrows=True, bounce=False,
                    arrow_fracs=(0.2, 0.8), force_single=False):
    """Overlay a trajectory in the current axes: line (optionally labelled for
    the legend), start marker (o), and end marker (s, the equilibrium). x, y are
    1-D arrays ordered in time (equal time steps, so arc length per step measures
    speed).

    arrows       : add direction arrows -- double chevron where the flow is fast,
        single where slow.
    bounce       : mark an overshoot peak with a ceiling + downward arrow (used
        in the (B, R) panel, where the fixed-B path retraces itself).
    force_single : draw single-headed arrows regardless of speed (used in the
        (B, R) panel, where a double chevron clutters the retraced line)."""
    plt.plot(x, y, '-', color=colours[1], label=label)
    plt.plot(x[0], y[0], 'o', color=colours[1])
    plt.plot(x[-1], y[-1], 's', color=colours[1])

    if bounce:
        for i in turning_point_indices(y):
            draw_bounce(x[i], y[i])

    if arrows:
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        if s[-1] == 0:
            return
        speed = np.diff(s)                   # arc length per (constant) time step
        fast = 0.2 * speed.max()             # threshold separating fast/slow flow
        for f in arrow_fracs:
            i = min(np.searchsorted(s, f * s[-1]), len(speed) - 1)
            n_heads = 1 if force_single else (2 if speed[i] > fast else 1)
            _draw_arrow(x, y, s, f, n_heads)


def plot_trajectories(t, y_tim, B_arr, y_s, y_u, Bmax_bifdiag):
    """4-panel figure: equilibrium branch (stable/unstable) with the trajectory
    overlaid in the (B, R), (B, rho) and (R, rho) planes, plus B(t)."""
    fig = plt.figure(figsize=(2, 2))   # grid units, not inches: show() scales by wunit/hunit per subplot

    ax1 = plt.subplot(2, 2, 1)                # Rac vs B
    plt.xlim(0, Bmax_bifdiag); plt.ylim(0, 1)   # set first so draw_bounce can size itself
    plt.plot(B_arr, y_s[:, 0], '-', color=colours[0], label='stable')
    plt.plot(B_arr, y_u[:, 0], '--', color=colours[0], label='unstable')
    # fixed-B path retraces itself: single up-arrow on the ascent, then a bounce
    draw_trajectory(y_tim[2, :], y_tim[0, :], label='trajectory', bounce=True,
                    arrow_fracs=(0.08,), force_single=True)
    plt.xlabel('B'); plt.ylabel('Rac')

    plt.subplot(2, 2, 2)                      # rho vs B
    plt.plot(B_arr, y_s[:, 1], '-', color=colours[0])
    plt.plot(B_arr, y_u[:, 1], '--', color=colours[0])
    draw_trajectory(y_tim[2, :], y_tim[1, :])
    plt.xlabel('B'); plt.ylabel('rho'); plt.ylim(0, 1); plt.xlim(0, Bmax_bifdiag)

    plt.subplot(2, 2, 3)                      # Rac vs rho (R on vertical axis, like panel 1)
    plt.plot(y_s[:, 1], y_s[:, 0], '-', color=colours[0])
    plt.plot(y_u[:, 1], y_u[:, 0], '--', color=colours[0])
    draw_trajectory(y_tim[1, :], y_tim[0, :])
    plt.xlabel('rho'); plt.ylabel('Rac'); plt.ylim(0, 1); plt.xlim(0, 1)

    plt.subplot(2, 2, 4)                      # B vs t
    plt.plot(t, y_tim[2, :], '-', color=colours[1])
    plt.xlabel('t'); plt.ylabel('B')
    plt.ylim(0, Bmax_bifdiag)                 # same B range as the bifurcation panels

    # one shared legend above the panels (stable / unstable / trajectory apply to all)
    fig.suptitle(' ')   # reserves the top margin that plottingtools.show() leaves for a title
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=10)


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
