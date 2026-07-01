import numpy as np
from scipy.integrate import solve_ivp

import LimitCycle as lc


def critical_eigenvector(M):
    w, V = np.linalg.eig(M)
    i = np.argmin(np.abs(w + 1.0))
    v = np.real(V[:, i])
    return v / np.linalg.norm(v)


def _state_at(y0, T, p1, xi):
    sol = solve_ivp(lambda t, y: T * lc.vector_field(y, p1), [0, xi], y0,
                    method='Radau', rtol=lc.RTOL, atol=lc.ATOL)
    return sol.y[:, -1]


def is_period_two(y0, T, p1, tol=1e-3):
    half = _state_at(y0, T, p1, 0.5)
    yf, _ = lc.integrate_period(y0, T, p1)
    return (np.linalg.norm(half - y0) > tol) and (np.linalg.norm(yf - y0) < 1e-6)


def branch_switch(pd, eps=2e-2, deltas=(0.002, 0.005, 0.01, 0.02)):
    y0c, Tc, p1c, M = pd['y0'], pd['T'], pd['p1'], pd['M']
    pert = eps * np.linalg.norm(y0c) * critical_eigenvector(M)
    for side in (+1, -1):
        for dp in deltas:
            p1_try = p1c + side * dp
            y0n, Tn, Mn, ok = lc.refine(p1_try, y0c + pert, 2.0 * Tc)
            if not ok:
                continue
            if abs(Tn - 2.0 * Tc) > 0.4 * 2.0 * Tc:
                continue
            if is_period_two(y0n, Tn, p1_try):
                return {'p1': p1_try, 'y0': y0n, 'T': Tn, 'M': Mn, 'side': side}
    return None


def summarize(level, pd):
    return {'level': level, 'period_factor': 2 ** (level - 1),
            'p1': pd['p1'], 'T': pd['T'], 'y0': pd['y0'],
            'mu': pd['mu'], 'nu0': pd['nu0'], 'delta': pd['delta']}


def find_cascade(pd1, max_levels=4, p1_max=1.0):
    results = [summarize(1, pd1)]
    branches = {}
    current = pd1
    for level in range(2, max_levels + 1):
        print(f"\n--- branch switch to period-{2 ** (level - 1)} orbit ---")
        bs = branch_switch(current)
        if bs is None:
            print("  branch switch failed; stopping cascade")
            break
        print(f"  seeded at p1 = {bs['p1']:.5f}, T = {bs['T']:.4f} "
              f"(side {bs['side']:+d})")
        mu_seed = current['mu'] ** 2          # carry the active label across switch
        history, mults, pd = lc.continue_in_p1(bs['p1'], bs['y0'], bs['T'],
                                               p1_max=p1_max, mu_seed=mu_seed)
        branches[level] = (history, mults)
        if pd is None:
            print(f"  no further period doubling found below p1 = {p1_max}")
            break
        crit = pd['mu'][np.argmin(np.abs(pd['mu'] + 1.0))]
        if pd['delta'] > 1e-6 or abs(crit + 1.0) > 1e-3 \
                or not (0.5 <= pd['p1'] <= 1.0):
            print(f"  crossing at p1 = {pd['p1']:.4f} is spurious "
                  f"(delta = {pd['delta']:.1e}); accuracy lost, stopping cascade")
            break
        results.append(summarize(level, pd))
        current = pd
    return results, branches


if __name__ == "__main__":
    pd_file = 'PD_Point.npy'
    try:
        rec = np.load(pd_file)
        y0c, Tc, p1c = rec[:3], rec[3], rec[4]
        _, Mc = lc.integrate_period(y0c, Tc, p1c)
        print(f"Loaded first period doubling from {pd_file}: "
              f"p1 = {p1c:.5f}, T = {Tc:.4f}")
    except FileNotFoundError:
        print("PD_Point.npy not found; locating the first period doubling ...")
        y0, T, _, _ = lc.seed_cycle(0.74)
        _, _, pd = lc.continue_in_p1(0.74, y0, T)
        y0c, Tc, p1c, Mc = pd['y0'], pd['T'], pd['p1'], pd['M']

    _, nu0c, deltac = lc.floquet(Mc, y0c, p1c)
    pd1 = {'p1': p1c, 'y0': y0c, 'T': Tc, 'M': Mc,
           'mu': lc.order_multipliers(Mc, y0c, p1c), 'nu0': nu0c,
           'delta': deltac}
    results, branches = find_cascade(pd1)

    print("\n=== Period-doubling cascade ===")
    print(f"{'level':>5} | {'period x':>8} | {'p1*':>9} | {'T':>9} | "
          f"{'crit. mult.':>12} | {'delta':>9}")
    for r in results:
        crit = r['mu'][np.argmin(np.abs(r['mu'] + 1.0))]
        print(f"{r['level']:>5} | {r['period_factor']:>8} | {r['p1']:>9.5f} | "
              f"{r['T']:>9.4f} | {crit.real:>12.5f} | {r['delta']:>9.2e}")

    if len(results) >= 3:
        d = ((results[-2]['p1'] - results[-3]['p1']) /
             (results[-1]['p1'] - results[-2]['p1']))
        print(f"\nFeigenbaum ratio (p1_n - p1_n-1)/(p1_n+1 - p1_n) = {d:.4f} "
              f"(theory 4.6692)")

    np.save('PeriodDoubling_Results.npy', np.array(results, dtype=object))
    for level, (hist, mlt) in branches.items():
        np.save(f'PD_Branch_{2 ** (level - 1)}.npy', hist)
        np.save(f'PD_Mults_{2 ** (level - 1)}.npy', mlt)
    print(f"\nSaved {len(results)} bifurcation(s) to PeriodDoubling_Results.npy "
          f"and {len(branches)} branch table(s).")
