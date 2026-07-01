import numpy as np
from scipy.integrate import solve_ivp

P2, P3, P4, P5, P6, P7, P8 = 2.5, 0.6, 1.5, 4.5, 1.0, 0.2, 0.5

RTOL, ATOL = 1e-9, 1e-11
T_SCALE = 20.0


def vector_field(y, p1):
    x1, x2, x3 = y
    dx1 = x1 * (1 - x1) - p1 * x1 * x2 - P2 * x1 * x3
    dx2 = P3 * x2 * (1 - x2) - P4 * x1 * x2
    dx3 = (P5 * x1 * x3) / (x1 + P6) - P7 * x1 * x3 - P8 * x3
    return np.array([dx1, dx2, dx3])


def jacobian(y, p1):
    x1, x2, x3 = y
    J = np.empty((3, 3))
    J[0, 0] = 1 - 2 * x1 - p1 * x2 - P2 * x3
    J[0, 1] = -p1 * x1
    J[0, 2] = -P2 * x1
    J[1, 0] = -P4 * x2
    J[1, 1] = P3 * (1 - 2 * x2) - P4 * x1
    J[1, 2] = 0.0
    J[2, 0] = P5 * P6 * x3 / (x1 + P6) ** 2 - P7 * x3
    J[2, 1] = 0.0
    J[2, 2] = P5 * x1 / (x1 + P6) - P7 * x1 - P8
    return J


def _augmented_rhs(xi, s, T, p1):
    y = s[:3]
    Phi = s[3:].reshape(3, 3)
    dy = T * vector_field(y, p1)
    dPhi = T * (jacobian(y, p1) @ Phi)
    return np.concatenate([dy, dPhi.ravel()])


def integrate_period(y0, T, p1):
    s0 = np.concatenate([y0, np.eye(3).ravel()])
    sol = solve_ivp(_augmented_rhs, [0, 1], s0, args=(T, p1),
                    method='Radau', rtol=RTOL, atol=ATOL)
    yf = sol.y[:3, -1]
    M = sol.y[3:, -1].reshape(3, 3)
    return yf, M


def floquet(M, y0=None, p1=None):
    mu, V = np.linalg.eig(M)
    if y0 is not None:
        f = vector_field(y0, p1)
        f = f / np.linalg.norm(f)
        score = np.abs(np.conj(V).T @ f) / np.linalg.norm(V, axis=0)
        i0 = int(np.argmax(score))
    else:
        i0 = int(np.argmin(np.abs(mu - 1.0)))
    nu0 = mu[i0]
    return mu, nu0, abs(1.0 - nu0)


def order_multipliers(M, y0, p1, mu_prev=None):
    mu, V = np.linalg.eig(M)
    f = vector_field(y0, p1)
    f = f / np.linalg.norm(f)
    score = np.abs(np.conj(V).T @ f) / np.linalg.norm(V, axis=0)
    i0 = int(np.argmax(score))                      # trivial: eigenvector ∥ flow
    rest = [k for k in range(len(mu)) if k != i0]
    if mu_prev is None:
        rest.sort(key=lambda k: -abs(mu[k]))         # slot 1 = active (|nu|->1)
    else:
        ordered, used = [], set()
        for pm in mu_prev[1:]:                      # continuity: match to last step
            k = min((kk for kk in rest if kk not in used),
                    key=lambda kk: abs(mu[kk] - pm))
            ordered.append(k)
            used.add(k)
        ordered += [k for k in rest if k not in used]
        rest = ordered
    return mu[[i0] + rest]


def pd_test(M):
    return np.linalg.det(M + np.eye(3))


def _residual_jac(u, y_anchor, with_jac=True):
    y0, T, p1 = u[:3], u[3], u[4]
    yf, M = integrate_period(y0, T, p1)
    v = vector_field(y_anchor, p1)

    R = np.empty(4)
    R[:3] = yf - y0
    R[3] = v @ (y0 - y_anchor)
    if not with_jac:
        return R, None, M

    J = np.zeros((4, 5))
    J[:3, :3] = M - np.eye(3)
    J[:3, 3] = vector_field(yf, p1)
    J[3, :3] = v

    eps = 1e-7
    yf_p, _ = integrate_period(y0, T, p1 + eps)
    v_p = vector_field(y_anchor, p1 + eps)
    R_p = np.append(yf_p - y0, v_p @ (y0 - y_anchor))
    J[:, 4] = (R_p - R) / eps
    return R, J, M


def refine(p1, y0, T, y_anchor=None, tol=1e-11, max_iter=30):
    if y_anchor is None:
        y_anchor = y0.copy()
    u = np.append(np.asarray(y0, float), float(T))
    for _ in range(max_iter):
        R, J, M = _residual_jac(np.append(u, p1), y_anchor)
        if np.linalg.norm(R) < tol:
            return u[:3], u[3], M, True
        step = np.linalg.lstsq(J[:, :4], -R, rcond=None)[0]
        u += step
    R, _, M = _residual_jac(np.append(u, p1), y_anchor, with_jac=False)
    return u[:3], u[3], M, np.linalg.norm(R) < 1e-7


def estimate_period(t, x1):
    interior = (x1[1:-1] > x1[:-2]) & (x1[1:-1] > x1[2:])
    peaks = t[1:-1][interior]
    return float(np.median(np.diff(peaks))) if peaks.size >= 2 else 17.5


def seed_cycle(p1, x_init=(0.10, 0.70, 0.15), t_max=800.0):
    t_eval = np.linspace(0, t_max, 40000)
    sol = solve_ivp(lambda t, y: vector_field(y, p1), [0, t_max], list(x_init),
                    t_eval=t_eval, rtol=1e-9, atol=1e-11, method='Radau')
    tail = sol.t > 0.6 * t_max
    T0 = estimate_period(sol.t[tail], sol.y[0, tail])
    return refine(p1, sol.y[:, -1], T0)


def _tangent(u, y_anchor, w, prev=None):
    _, J, _ = _residual_jac(u, y_anchor)
    t = np.linalg.svd(J)[2][-1]
    if prev is not None:
        if np.dot(t, prev) < 0:
            t = -t
    elif t[4] < 0:
        t = -t
    return t / np.sqrt(np.sum(w * t * t))


def _localize_pd(p1_lo, test_lo, p1_hi, test_hi, y0, T, max_iter=40):
    for _ in range(max_iter):
        denom = test_hi - test_lo
        p1_m = 0.5 * (p1_lo + p1_hi) if denom == 0 else \
            p1_lo - test_lo * (p1_hi - p1_lo) / denom
        p1_m = min(max(p1_m, p1_lo), p1_hi)
        y0, T, M, _ = refine(p1_m, y0, T)
        test_m = pd_test(M)
        if abs(test_m) < 1e-10 or (p1_hi - p1_lo) < 1e-10:
            return p1_m, y0, T, M
        if (test_lo < 0) != (test_m < 0):
            p1_hi, test_hi = p1_m, test_m
        else:
            p1_lo, test_lo = p1_m, test_m
    return p1_m, y0, T, M


def continue_in_p1(p1_0, y0_0, T_0, ds=0.02, p1_max=1.0, max_steps=2000,
                   mu_seed=None, verbose=True):
    w = np.array([1.0, 1.0, 1.0, 1.0 / T_SCALE ** 2, 1.0])
    u = np.append(np.append(y0_0, T_0), p1_0)
    y_anchor = u[:3].copy()
    tau = _tangent(u, y_anchor, w)

    _, _, M = _residual_jac(u, y_anchor, with_jac=False)
    test_prev = pd_test(M)
    _, _, delta = floquet(M, u[:3], u[4])
    mu_track = order_multipliers(M, u[:3], u[4], mu_seed)
    history = [(p1_0, T_0, delta, test_prev)]
    mults = [mu_track]

    if verbose:
        print(f"{'p1':>8} | {'T':>8} | {'delta':>9} | {'det(M+I)':>11} | "
              f"{'active mult.':>13}")
        print(f"{p1_0:8.5f} | {T_0:8.3f} | {delta:9.2e} | {test_prev:11.3e} | "
              f"{mu_track[1].real:13.5f}")

    for _ in range(max_steps):
        if u[4] >= p1_max:
            break
        u_pred = u + ds * tau
        un = u_pred.copy()
        ok = False
        for it in range(20):
            R, J, M = _residual_jac(un, y_anchor)
            g = np.sum(w * (un - u) * tau) - ds
            step = np.linalg.solve(np.vstack([J, w * tau]),
                                   -np.append(R, g))
            un += step
            if np.linalg.norm(step) < 1e-10:
                ok = True
                break
        if not ok:
            ds *= 0.5
            if ds < 1e-6:
                break
            continue

        test = pd_test(M)
        _, _, delta = floquet(M, un[:3], un[4])
        mu_track = order_multipliers(M, un[:3], un[4], mu_track)
        history.append((un[4], un[3], delta, test))
        mults.append(mu_track)
        if verbose:
            print(f"{un[4]:8.5f} | {un[3]:8.3f} | {delta:9.2e} | {test:11.3e} | "
                  f"{mu_track[1].real:13.5f}")

        if (test_prev < 0) != (test < 0):
            p1_pd, y0_pd, T_pd, M_pd = _localize_pd(
                u[4], test_prev, un[4], test, un[:3], un[3])
            mu, nu0, delta_pd = floquet(M_pd, y0_pd, p1_pd)
            mu_track = order_multipliers(M_pd, y0_pd, p1_pd, mu_track)
            mults.append(mu_track)
            return (np.array(history), np.array(mults),
                    {'p1': p1_pd, 'y0': y0_pd, 'T': T_pd, 'M': M_pd,
                     'mu': mu_track, 'nu0': nu0, 'delta': delta_pd})

        tau = _tangent(un, un[:3], w, prev=tau)
        u, y_anchor, test_prev = un, un[:3].copy(), test
        ds = min(ds * 1.3, 0.05) if it < 4 else ds

    return np.array(history), np.array(mults), None


def plot_floquet(history, mults, pd, path='../figures/floquet_period1.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p1 = history[:, 0]
    mu = mults[:len(history)]
    mask = p1 <= pd['p1'] + 1e-9
    p1m = np.append(p1[mask], pd['p1'])
    mum = np.vstack([mu[mask], mults[-1]])
    order = np.argsort(p1m)
    p1m, mum = p1m[order], mum[order]

    # the two non-trivial multipliers cross near nu=0; un-swap the tracked
    # slots after the crossing so each curve follows one physical mode
    nt = mum[:, 1:].copy()
    ic = int(np.argmin(np.abs(nt[:, 0] - nt[:, 1])))
    nt[ic + 1:] = nt[ic + 1:][:, ::-1]
    mum[:, 1:] = nt

    re, mod = mum.real, np.abs(mum)
    labels = [r'$\nu_0$ (trivial, $\equiv 1$)',
              r'$\nu_1$ (period-doubling mode)', r'$\nu_2$ (contracting)']
    colors = ['black', 'tab:red', 'tab:blue']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in range(3):
        ax1.plot(p1m, re[:, k], color=colors[k], label=labels[k])
    ax1.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax1.axhline(-1.0, color='gray', ls='--', lw=0.8)
    ax1.axvline(pd['p1'], color='orange', ls='-.', lw=1.0)
    ax1.plot(pd['p1'], -1.0, 'o', mfc='none', mec='orange', ms=9)
    ax1.set_xlabel(r'$p_1$')
    ax1.set_ylabel(r'$\mathrm{Re}(\nu_k)$')
    ax1.set_title('Floquet multipliers (real part)')
    ax1.legend(loc='center left')
    ax1.grid(alpha=0.3)

    for k in range(3):
        ax2.plot(p1m, mod[:, k], color=colors[k], label=labels[k])
    ax2.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax2.axvline(pd['p1'], color='orange', ls='-.', lw=1.0)
    ax2.set_xlabel(r'$p_1$')
    ax2.set_ylabel(r'$|\nu_k|$')
    ax2.set_title('Floquet multiplier modulus (stability)')
    ax2.grid(alpha=0.3)

    fig.suptitle(r'Limit cycle Floquet multipliers up to first period '
                 r'doubling ($\tilde p_1 = %.4f$)' % pd['p1'])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


if __name__ == "__main__":
    P1_START = 0.74

    print(f"Seeding limit cycle at p1 = {P1_START} ...")
    y0, T, M, ok = seed_cycle(P1_START)
    _, _, delta = floquet(M, y0, P1_START)
    print(f"  seed: T = {T:.4f}, |1 - nu0| = {delta:.2e}, converged = {ok}\n")

    history, mults, pd = continue_in_p1(P1_START, y0, T)
    np.save('LimitCycle_Data.npy', history)
    np.save('LimitCycle_Mults.npy', mults)

    if pd is not None:
        print("\n=== Period-doubling bifurcation found ===")
        print(f"  p1*    = {pd['p1']:.5f}")
        print(f"  period = {pd['T']:.4f}")
        print(f"  Floquet multipliers = {np.round(pd['mu'], 5)}")
        print(f"  trivial nu0 = {pd['nu0']:.6f}  (delta = {pd['delta']:.2e})")
        np.save('PD_Point.npy', np.concatenate([pd['y0'], [pd['T'], pd['p1']]]))
        plot_floquet(history, mults, pd)
    else:
        print(f"\nNo period doubling detected on [{P1_START}, {p1_max}]")
