import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plottingtools as st

from continuation import fun, jac, fun_ext, fun_ext_sat, run_continuation
from timeintegrator import time_integrator
from rootfinding import newton_small

mpl.use('TkAgg')
colours = st.configure()
import plottingtools.step_config as cf   # panel size / dpi constants used by show()


def save_house(fig, fname, ncols, nrows, top=None):
    """Save `fig` at the house panel size (cf.wunit x cf.hunit per panel) and the
    configured save dpi, then restore the grid-unit size so plottingtools.show()
    rescales it identically for the interactive display. `top` reserves a strip at
    the top of the figure for a figure-level legend."""
    fig.set_size_inches(ncols * cf.wunit, nrows * cf.hunit)
    fig.tight_layout(rect=[0, 0, 1, top]) if top else fig.tight_layout()
    fig.savefig(fname, dpi=cf.savedpi, bbox_inches='tight')
    fig.set_size_inches(nrows, ncols)


# ============================================================================
# Functions
# ============================================================================
def integrate_trajectory(params, ic, dt, T, eps, Bmax, rhs=fun_ext):
    """RK4-integrate the extended (R, rho, B) system.

    params : (A, R0, rho0, deltaR, n)
    ic     : initial condition (R, rho, B)
    eps    : rate of change of B (eps = 0 keeps B fixed at ic[2])
    rhs    : extended RHS -- fun_ext (dB/dt=eps, Q3) or fun_ext_sat
             (dB/dt=eps*(Bmax-B)/Bmax, Q4).
    Returns t and y_tim of shape (3, len(t)) with rows [R, rho, B].
    """
    params_ext = (*params, Bmax, eps)
    return time_integrator(rhs, params_ext, ic, dt, T)


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
# Exercise 2 -- domains of attraction
# ============================================================================
def equilibria_at_B(B_target, B_arr, y_arr):
    """Equilibria of the (R, rho) system at fixed B, read off the continuation
    branch: the S-shaped branch crosses B = B_target wherever (B_arr - B_target)
    changes sign. Linearly interpolate the (R, rho) equilibrium at each crossing.
    Returns an (m, 2) array of equilibria (m = 3 at the default B = 0.03)."""
    g = B_arr - B_target
    cross = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    eqs = []
    for i in cross:
        w = g[i] / (g[i] - g[i + 1])             # 0..1 along the segment
        eqs.append(y_arr[i] + w * (y_arr[i + 1] - y_arr[i]))
    return np.asarray(eqs)


def classify_equilibria(eqs, B, params):
    """Split equilibria into stable nodes and saddles by the sign of the largest
    real part of the Jacobian eigenvalues. Returns (stable, saddles), each a list
    of (R, rho) arrays."""
    stable, saddles = [], []
    for y in eqs:
        ev = np.linalg.eigvals(jac(y, B, params))
        (stable if np.max(np.real(ev)) < 0 else saddles).append(y)
    return stable, saddles


def print_eigenvalues(eqs, B, params):
    """Tabulate each fixed point and its Jacobian eigenvalues at fixed B. Both
    eigenvalues are real here; two negatives -> stable node, opposite signs ->
    saddle. The smallest |eigenvalue| sets the slowest relaxation timescale."""
    print(f"\nFixed points and Jacobian eigenvalues at B={B}:")
    print(f"{'(Rac, rho)':>18}   {'eig 1':>9}  {'eig 2':>9}   type")
    for y in sorted(eqs, key=lambda p: -p[0]):
        ev = np.sort(np.linalg.eigvals(jac(y, B, params)).real)[::-1]
        typ = 'stable node' if ev.max() < 0 else ('saddle' if ev.min() < 0 else 'unstable')
        print(f"  ({y[0]:6.4f}, {y[1]:6.4f})   {ev[0]:9.4f}  {ev[1]:9.4f}   {typ}")


def basin_map(params, B, stable_eqs, ngrid=350, dt=0.05, T=50):
    """Brute-force domains of attraction at fixed B. Integrate a full grid of
    initial conditions over (R, rho) in [0, 1]^2 forward in time (RK4, vectorised
    over the whole grid at once) and label each start by the stable equilibrium it
    ends nearest. Returns RR, PP (the R, rho mesh) and an integer label array."""
    A, R0, rho0, deltaR, n = params

    def f(R, rho):                               # vectorised RHS at fixed B
        dR = A / (rho0**n + rho**n) * (1 - R) - deltaR * R
        drho = B / (R0**n + R**n) * (1 - rho) - rho
        return dR, drho

    r = np.linspace(0, 1, ngrid)
    RR, PP = np.meshgrid(r, r)                    # RR = Rac (x), PP = rho (y)
    R, rho = RR.copy(), PP.copy()
    for _ in range(int(T / dt)):                 # classic RK4 step on the arrays
        k1R, k1p = f(R, rho)
        k2R, k2p = f(R + 0.5 * dt * k1R, rho + 0.5 * dt * k1p)
        k3R, k3p = f(R + 0.5 * dt * k2R, rho + 0.5 * dt * k2p)
        k4R, k4p = f(R + dt * k3R, rho + dt * k3p)
        R = R + dt / 6 * (k1R + 2 * k2R + 2 * k3R + k4R)
        rho = rho + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)

    dist = np.stack([np.hypot(R - s[0], rho - s[1]) for s in stable_eqs])
    return RR, PP, np.argmin(dist, axis=0)


def trace_manifold(saddle, params, B, kind, ds=1e-6, dt=0.005, T=40):
    """Trace one invariant manifold of the saddle. Seed two points just either
    side of the saddle along the relevant eigenvector and integrate; index 0 of
    each returned branch sits at the saddle, increasing index runs outward.

    kind='stable'   : eigenvector of the most negative eigenvalue, integrated
        *backward* in time (y' = -f). This is the separatrix / basin boundary;
        the actual flow on it points toward the saddle.
    kind='unstable' : eigenvector of the most positive eigenvalue, integrated
        forward. Its two branches end on the two stable nodes, so it connects
        all three equilibria; the flow on it points away from the saddle.
    Returns a list of two (2, N) branch arrays."""
    w, V = np.linalg.eig(jac(saddle, B, params))
    pick = np.argmin if kind == 'stable' else np.argmax
    v = np.real(V[:, int(pick(np.real(w)))])
    v = v / np.linalg.norm(v)
    sign = -1.0 if kind == 'stable' else 1.0          # backward flow for the stable manifold
    rhs = lambda y, _p: sign * fun(y, B, params)
    branches = []
    for sgn in (1, -1):
        _, y = time_integrator(rhs, None, saddle + sgn * ds * v, dt, T)
        inside = (y[0] >= 0) & (y[0] <= 1) & (y[1] >= 0) & (y[1] <= 1)
        branches.append(y[:, inside])
    return branches


def _manifold_arrow(branch, frac, toward_saddle, color, dfrac=0.06, scale=20):
    """Draw one arrowhead a fraction `frac` along a manifold branch, measured by
    *arc length* (so it sits geometrically mid-curve, not bunched near the saddle
    where backward integration crowds the samples). Index 0 of the branch lies at
    the saddle; toward_saddle -> the head points back toward the saddle."""
    x, y = branch[0], branch[1]
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    if s[-1] == 0:
        return
    lo = int(np.clip(np.searchsorted(s, (frac - dfrac) * s[-1]), 0, len(x) - 1))
    hi = int(np.clip(np.searchsorted(s, (frac + dfrac) * s[-1]), lo + 1, len(x) - 1))
    near, far = branch[:, lo], branch[:, hi]          # lo is closer to the saddle
    xy, xytext = (near, far) if toward_saddle else (far, near)
    plt.annotate('', xy=xy, xytext=xytext,
                 arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4, mutation_scale=scale))


def plot_basins(RR, PP, labels, stable_eqs, saddles, separatrix, unstable, B):
    """(rho, Rac) phase plane shaded by domain of attraction -- rho on the
    horizontal, Rac on the vertical, matching the (rho, Rac) panel of Q1. Overlays
    the separatrix (stable manifold, arrows toward the saddle), the unstable
    manifold connecting the two nodes (dashed, arrows away) and all equilibria.
    RR, PP carry Rac and rho respectively; everything is plotted (rho, Rac)."""
    plt.figure(figsize=(1, 1))
    cmap = mpl.colors.ListedColormap(['#d6e6f4', '#d8efd8'])   # light blue / green
    plt.pcolormesh(PP, RR, labels, cmap=cmap, shading='nearest', rasterized=True)

    for br in separatrix:                         # stable manifold = basin boundary (solid)
        plt.plot(br[1], br[0], '-', color='k', lw=1.2)
        _manifold_arrow(br[::-1], 0.55, toward_saddle=True, color='k')
    for br in unstable:                           # unstable manifold = node-saddle-node (dashed)
        plt.plot(br[1], br[0], '--', color='k', lw=1.2)
        _manifold_arrow(br[::-1], 0.55, toward_saddle=False, color='k')

    for s in stable_eqs:
        plt.plot(s[1], s[0], 'o', color=colours[1], markersize=7)   # stable nodes
    for s in saddles:
        plt.plot(s[1], s[0], 'X', color=colours[3], markersize=8)   # saddle

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel('Rho'); plt.ylabel('Rac')


# ============================================================================
# Exercise 3a -- trajectories under a drifting B (dB/dt = eps)
# ============================================================================
def right_fold(B_arr):
    """B value of the right saddle-node (fold): the first point where the
    pseudo-arclength branch turns back in B (B starts to decrease)."""
    dec = np.where(np.diff(B_arr) < 0)[0]
    return B_arr[dec[0]] if len(dec) else np.nan


def plot_drift(B_arr, y_s, y_u, trajectories, eps_list, B_plot, Bfold):
    """Two bifurcation diagrams, (B, Rac) and (B, rho), with one drifting-B
    trajectory per eps overlaid on the stable/unstable equilibrium branches. The
    dotted vertical line marks the right fold. A slow eps tracks the upper branch
    and tips at the fold; a fast eps lags and tips well beyond it."""
    tcols = [colours[1], colours[2], colours[3]]
    fig = plt.figure(figsize=(1, 2))                 # 1 row x 2 cols (grid units)
    for k, name in ((0, 'Rac'), (1, 'rho')):
        plt.subplot(1, 2, k + 1)
        plt.axvline(Bfold, color='grey', ls=':', lw=0.8)
        plt.plot(B_arr, y_s[:, k], '-', color=colours[0], label='stable' if k == 0 else None)
        plt.plot(B_arr, y_u[:, k], '--', color=colours[0], label='unstable' if k == 0 else None)
        for y, eps, c in zip(trajectories, eps_list, tcols):
            plt.plot(y[2], y[k], '-', color=c, lw=1.3,
                     label=(rf'$\epsilon$={eps:g}' if k == 0 else None))
            plt.plot(y[2, 0], y[k, 0], 'o', color=c, ms=4)
        plt.xlabel('B'); plt.ylabel(name); plt.xlim(0, B_plot); plt.ylim(0, 1)
    fig.suptitle(' ')                                # reserve top margin for the legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=9)
    return fig


# ============================================================================
# Exercise 3b -- return time, eigenvalues and critical slowing down
# ============================================================================
def slow_mode(eq, B, params):
    """Unit eigenvector of the least-stable (slowest) eigenvalue at the equilibrium
    eq, oriented with a positive rho-component (so a kick pushes rho up), together
    with |lambda| of that mode. The slow mode is the one whose eigenvalue -> 0 at
    the fold, i.e. the direction that exhibits critical slowing down."""
    w, V = np.linalg.eig(jac(eq, B, params))
    k = int(np.argmax(w.real))
    v = V[:, k].real
    v = v / np.linalg.norm(v)
    return (v if v[1] > 0 else -v), abs(w[k].real)


def recovery_rate(params, B, kick=0.01):
    """Measured exponential return rate of a small state perturbation at fixed B.
    Sit at the high-Rac equilibrium, kick along the slow eigenvector, integrate
    with B frozen (eps=0) and fit the slope of log|deviation| over the late
    window. Should reproduce |lambda_max(B)| from the linearised system."""
    eq = newton_small(fun, jac, np.array([0.8, 0.05]), B, params)
    v, lam = slow_mode(eq, B, params)
    Trun = max(60.0, 12.0 / lam)                     # scale the window with 1/|lambda|
    t, y = integrate_trajectory(params, (eq[0] + kick * v[0], eq[1] + kick * v[1], B),
                                0.005, Trun, 0.0, 1.0)
    dev = np.hypot(y[0] - eq[0], y[1] - eq[1])
    ok = dev > 1e-10
    tt, dd = t[ok], dev[ok]
    win = (tt > 0.4 * tt[-1]) & (tt < 0.9 * tt[-1])  # skip the fast transient and the noise floor
    return -np.polyfit(tt[win], np.log(dd[win]), 1)[0]


def plot_csd(B_branch, rate_theory, Bs, rates_meas, Bfold):
    """(a) Critical slowing down at the fold: measured recovery rate and return
    time (markers) versus the theoretical |lambda_max| of the high-Rac branch
    (line). The rate -> 0 and the return time -> infinity as B -> Bfold."""
    fig = plt.figure(figsize=(1, 2))
    plt.subplot(1, 2, 1)
    plt.axvline(Bfold, color='grey', ls=':', lw=0.8)
    plt.plot(B_branch, rate_theory, '-', color=colours[0])   # theory |lambda_max|
    plt.plot(Bs, rates_meas, 'o', color=colours[3])          # measured (perturb + fit)
    plt.xlabel('B'); plt.ylabel(r'recovery rate $|\lambda_{max}|$')
    plt.xlim(0, 0.055); plt.ylim(0, 1.1)
    plt.subplot(1, 2, 2)
    plt.axvline(Bfold, color='grey', ls=':', lw=0.8)
    sel = rate_theory > 0.04                          # drop the near-zero tail (1/rate blows up)
    plt.plot(B_branch[sel], 1 / rate_theory[sel], '-', color=colours[0])
    plt.plot(Bs, 1 / rates_meas, 'o', color=colours[3])
    plt.xlabel('B'); plt.ylabel(r'return time $1/|\lambda_{max}|$')
    plt.xlim(0, 0.055); plt.ylim(0, 30)
    return fig


def plot_perturbations(params, B_arr, y_s, y_u, ic_eq, B0, Bfold,
                       k=1, ylabel='rho', ylims=((0, 0.30), (0, 1.0))):
    """(b) State perturbations along the drifting trajectory, slow vs fast eps,
    shown as component k vs B (k=1 -> rho, k=0 -> Rac; ylims set the per-eps zoom).
    For each eps the unperturbed drift starts at the high-Rac equilibrium; at
    selected B the state is kicked along the slow eigenvector and integrated on.
    Slow eps: the recovery excursions broaden as B -> fold (critical slowing down).
    Fast eps: perturbations are swept along and the system tips before recovering."""
    def run(eps, Bps, Bend, delta):
        Trun = (Bend - B0) / eps
        h = min(0.01, Trun / 12000)
        _, yu = integrate_trajectory(params, (ic_eq[0], ic_eq[1], B0), h, Trun, eps, Bend)
        perts = []
        for Bp in Bps:
            i = int(np.argmin(np.abs(yu[2] - Bp)))
            st = yu[:, i].copy()
            v, _ = slow_mode(st[:2], st[2], params)
            _, yp = integrate_trajectory(params, (st[0] + delta * v[0], st[1] + delta * v[1], st[2]),
                                         h, Trun - i * h, eps, Bend)
            perts.append(yp)
        return yu, perts

    fig = plt.figure(figsize=(1, 2))
    cfg = [(1e-3, [0.02, 0.035, 0.045, 0.050], 0.07, 0.05, (0, 0.065)),
           (1e-1, [0.005, 0.03, 0.08], 0.30, 0.05, (0, 0.30))]
    for s, (eps, Bps, Bend, delta, xl) in enumerate(cfg):
        yu, perts = run(eps, Bps, Bend, delta)
        plt.subplot(1, 2, s + 1)
        plt.plot(B_arr, y_s[:, k], '-', color=colours[0], lw=1)
        plt.plot(B_arr, y_u[:, k], '--', color=colours[0], lw=1)
        for yp in perts:
            plt.plot(yp[2], yp[k], '-', color=colours[3], lw=1.1)   # perturbed runs
        plt.plot(yu[2], yu[k], '-', color=colours[4], lw=1.6)       # unperturbed drift
        plt.axvline(Bfold, color='grey', ls=':', lw=0.8)
        plt.xlabel('B'); plt.ylabel(ylabel); plt.xlim(*xl); plt.ylim(*ylims[s])
        plt.title(rf'$\epsilon$={eps:g}', fontsize=10)
    return fig


# ============================================================================
# Exercise 4 -- rate-induced tipping (dB/dt = eps*(Bmax-B)/Bmax, B saturates at Bmax)
# ============================================================================
def integrate_saturating(params, ic, eps, Bmax, dt=0.004, relax=80.0):
    """Integrate the (R, rho, B) system under the saturating law until B has
    reached Bmax and the state has settled. Because dB/dt -> 0 as B -> Bmax the
    parameter freezes on its own; we run for the exponential approach time
    (Bmax/eps)*ln(...) plus a relaxation tail. Returns t, y (rows R, rho, B)."""
    B0 = ic[2]
    T = (Bmax / eps) * np.log((Bmax - B0) / 1e-7) + relax
    return integrate_trajectory(params, ic, dt, T, eps, Bmax, rhs=fun_ext_sat)


def final_rac(params, ic, eps, Bmax, dt=0.01):
    """Settled Rac after the saturating drift -- the diagnostic that is binary
    here: ~0.81 (tracked, Y+) or ~0.006 (tipped, Z+)."""
    return integrate_saturating(params, ic, eps, Bmax, dt=dt, relax=60.0)[1][0, -1]


def critical_rate(params, ic, Bmax, lo=1e-3, hi=1.0, tol=1e-4, thresh=0.4):
    """Critical rate eps_c by bisection on the monotone tracked/tipped outcome
    (final Rac < thresh). Below eps_c the orbit end-point-tracks Y- -> Y+; above
    it is overtaken by the moving separatrix and lands on Z+."""
    tip = lambda e: final_rac(params, ic, e, Bmax) < thresh
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if tip(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def plot_rtip_diagram(params, ic, Bmax_list, eps_grid, eps_c):
    """R-tipping diagram: settled Rac versus the rate eps (log axis), one curve per
    Bmax. Bmax below the fold (0.04) gives a sharp step at eps_c -- tipping depends
    on the *rate* (rate-induced). Bmax above the fold tips for every rate, because
    the swept path crosses the saddle-node regardless of speed (bifurcation-induced)."""
    fig = plt.figure(figsize=(1, 1))
    cols = [colours[0], colours[3]]
    for Bmax, c in zip(Bmax_list, cols):
        finals = [final_rac(params, ic, e, Bmax) for e in eps_grid]
        plt.semilogx(eps_grid, finals, '-o', color=c, ms=3, lw=1.2)
        plt.text(eps_grid[1], finals[0] + 0.04, rf'$B_{{max}}$={Bmax:g}', color=c, fontsize=9)
    plt.axvline(eps_c, color='grey', ls=':', lw=0.8)
    plt.text(eps_c * 1.1, 0.5, rf'$\epsilon_c\approx{eps_c:.3f}$', color='grey',
             fontsize=9, rotation=90, va='center')
    plt.xlabel(r'rate $\epsilon$'); plt.ylabel('final Rac'); plt.ylim(0, 1)
    return fig


def plot_rtip_trajectories(params, ic, eps_pair, Bmax, B_arr, y_s, y_u, Bfold):
    """Sub- and super-critical drift trajectories on the (B, Rac) and (B, rho)
    bifurcation diagrams. The dashed line is Bmax, the dotted line the fold (Bmax
    sits below it). Slow eps climbs to the upper branch and tracks Y- -> Y+; fast
    eps lags, is overtaken by the moving separatrix at some B < Bmax, and drops to
    the lower branch (Z+) -- the orbit 'races off' before the path even ends."""
    cols = [colours[1], colours[3]]
    runs = [integrate_saturating(params, ic, e, Bmax) for e in eps_pair]
    fig = plt.figure(figsize=(1, 2))
    for k, name in ((0, 'Rac'), (1, 'rho')):
        plt.subplot(1, 2, k + 1)
        plt.plot(B_arr, y_s[:, k], '-', color=colours[0])
        plt.plot(B_arr, y_u[:, k], '--', color=colours[0])
        plt.axvline(Bfold, color='grey', ls=':', lw=0.8)
        plt.axvline(Bmax, color='grey', ls='--', lw=0.8)
        for (t, y), c in zip(runs, cols):
            plt.plot(y[2], y[k], '-', color=c, lw=1.3)
            plt.plot(y[2, 0], y[k, 0], 'o', color=c, ms=4)
            plt.plot(y[2, -1], y[k, -1], 's', color=c, ms=4)
        plt.xlabel('B'); plt.ylabel(name); plt.xlim(0, 0.06); plt.ylim(0, 1)
    return fig


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

    # --- Q1 plot / print ---
    print(f"endpoint: R={y_tim[0, -1]:.4f}, rho={y_tim[1, -1]:.4f}, B ={ic[-1]}")
    plot_trajectories(t, y_tim, B_arr, y_s, y_u, Bmax_bifdiag)
    save_house(plt.gcf(), '../figures/figure1.png', 2, 2, top=0.93)

    # --- Q2: domains of attraction at the default B ---
    B_q2 = 0.03
    eqs = equilibria_at_B(B_q2, B_arr, y_arr)
    stable_eqs, saddles = classify_equilibria(eqs, B_q2, params)
    print(f"B={B_q2}: {len(stable_eqs)} stable node(s), {len(saddles)} saddle(s)")
    print_eigenvalues([newton_small(fun, jac, e, B_q2, params) for e in eqs], B_q2, params)
    saddles = [newton_small(fun, jac, s, B_q2, params) for s in saddles]  # refine for the manifolds
    RR, PP, labels = basin_map(params, B_q2, stable_eqs)
    separatrix = [b for s in saddles for b in trace_manifold(s, params, B_q2, 'stable')]
    unstable = [b for s in saddles for b in trace_manifold(s, params, B_q2, 'unstable')]
    plot_basins(RR, PP, labels, stable_eqs, saddles, separatrix, unstable, B_q2)
    save_house(plt.gcf(), '../figures/figure2.png', 1, 1)

    # --- Q3: drifting B (dB/dt = eps). One continuation feeds 3a and 3b. ---
    Bmax_q3, B_plot, eps_list = 0.3, 0.25, (1e-3, 1e-2, 1e-1)
    Bq3, yq3, yq3_s, yq3_u, evs_q3, _ = run_continuation(params, B0_cont, Bmax_q3, ds)
    Bfold = right_fold(Bq3)
    ifold = int(np.where(np.diff(Bq3) < 0)[0][0])     # last index of the high-Rac branch
    ic_eq = yq3[0]                                     # start on the upper branch at B=B0_cont

    # Q3a: trajectories for slow vs fast drift, overlaid on the bifurcation diagrams
    drifts = [integrate_trajectory(params, (ic_eq[0], ic_eq[1], B0_cont), dt,
                                   (B_plot - B0_cont) / e, e, B_plot)[1] for e in eps_list]
    plot_drift(Bq3, yq3_s, yq3_u, drifts, eps_list, B_plot, Bfold)
    save_house(plt.gcf(), '../figures/figure3.png', 2, 1, top=0.88)

    # Q3b(a): critical slowing down -- measured recovery rate vs theory |lambda_max|
    Bs = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.048, 0.0505, 0.0512])
    rates = np.array([recovery_rate(params, B) for B in Bs])
    plot_csd(Bq3[:ifold + 1], -evs_q3[:ifold + 1], Bs, rates, Bfold)
    save_house(plt.gcf(), '../figures/figure4.png', 2, 1)

    # Q3b(b): state perturbations along the drift, slow (CSD) vs fast (tips first)
    plot_perturbations(params, Bq3, yq3_s, yq3_u, ic_eq, B0_cont, Bfold)               # rho(B)
    save_house(plt.gcf(), '../figures/figure5.png', 2, 1)
    plot_perturbations(params, Bq3, yq3_s, yq3_u, ic_eq, B0_cont, Bfold,               # Rac(B), to compare
                       k=0, ylabel='Rac', ylims=((0.55, 0.84), (0, 1.0)))
    save_house(plt.gcf(), '../figures/figure8.png', 2, 1)

    # --- Q4: rate-induced tipping. dB/dt = eps*(Bmax-B)/Bmax, Bmax below the fold. ---
    Bmax_q4 = 0.04                                  # endpoint, below the fold (B*~0.0514)
    ic_q4 = (0.5, 0.2, 0.01)                         # IC (R, rho, B0) from the assignment
    eps_c = critical_rate(params, ic_q4, Bmax_q4)
    print(f"Q4: critical rate eps_c = {eps_c:.4f} (eps_c/Bmax = {eps_c / Bmax_q4:.3f} /time)")

    # Q4(a): R-tipping diagram -- final Rac vs rate; threshold for Bmax<fold, always-tip for Bmax>fold
    eps_grid = np.logspace(-3, 0, 16)
    plot_rtip_diagram(params, ic_q4, [Bmax_q4, 0.07], eps_grid, eps_c)
    save_house(plt.gcf(), '../figures/figure6.png', 1, 1)

    # Q4(b): mechanism -- sub- vs super-critical drift on the bifurcation diagrams
    plot_rtip_trajectories(params, ic_q4, (0.03, 0.1), Bmax_q4, Bq3, yq3_s, yq3_u, Bfold)
    save_house(plt.gcf(), '../figures/figure7.png', 2, 1)

    st.show()
