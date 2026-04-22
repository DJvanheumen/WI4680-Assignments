import numpy as np
import matplotlib.pyplot as plt
import RootFinding as rf 

def run_continuation(start, end, steps):
    p_vals = np.linspace(start, end, steps)
    raw_data = []

    for p in p_vals:
        state = rf.get_coexistence_state(p)

        if not np.any(np.isnan(state)):
            J = rf.calculate_jacobian(*state, p)
            eigs = rf.compute_eigenvalues(J)
            raw_data.append({'p1': p, 'state': state, 'eigs': eigs})

    return raw_data

def process_results(data, tol):
    p = np.array([d['p1'] for d in data])
    states = np.array([d['state'] for d in data])
    eigs_matrix = np.array([d['eigs'] for d in data])
    
    eps_vals, im_parts, re_lead = [], [], []
    results = {}

    for eigs in eigs_matrix:
        val = (eigs[0] + eigs[1]) * (eigs[0] + eigs[2]) * (eigs[1] + eigs[2])
        eps_vals.append(np.real(val))
        idx_complex = np.argsort(np.abs(np.imag(eigs)))[::-1]
        re_lead.append(np.real(eigs[idx_complex[0]]))
        im_parts.append(np.abs(np.imag(eigs[idx_complex[0]])))

    eps, im_h, re_h = np.array(eps_vals), np.array(im_parts), np.array(re_lead)
    crossings = np.where(np.diff(np.sign(eps)))[0]
    lambda_tilde = p[crossings[0]] if len(crossings) > 0 else None
    
    criteria = [False, False, False]

    if lambda_tilde is not None:
        idx = crossings[0]
        criteria[0] = True # Existence is confirmed by the sign change in eps
    
        # Non-Degeneracy: 
        criteria[1] = im_h[idx] > tol and np.abs(re_h[idx]) < tol
    
        # Transversality: Slope check remains the same
        slope = (eps[idx+1] - eps[idx]) / (p[idx+1] - p[idx])
        criteria[2] = np.abs(slope) > tol

    results = {
        'p': p, 
        'x': states, 
        'eps': eps, 
        'im_hopf': im_h, 
        're_h': re_h,
        'lambda_tilde': lambda_tilde, 
        'hopf_flags': criteria}

    return results

def plot_with_stability(ax, x, y, stability_test, **kwargs):
    stable = stability_test <= 0

    ax.plot(np.where(stable, x, np.nan), np.where(stable, y, np.nan), linestyle='-', **kwargs)
    ax.plot(np.where(~stable, x, np.nan), np.where(~stable, y, np.nan), linestyle='--', **kwargs)

def plot_continuation_profile(proc):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    species_config = [
        {'data': proc['x'][:, 0], 'color': 'firebrick', 'label': r'Tumor ($x_1$)'},
        {'data': proc['x'][:, 1], 'color': 'seagreen', 'label': r'Healthy ($x_2$)'},
        {'data': proc['x'][:, 2], 'color': 'royalblue', 'label': r'Immune ($x_3$)'}
    ]

    for s in species_config:
        plot_with_stability(ax, proc['p'], s['data'], proc['re_h'], 
                            color=s['color'], label=s['label'], lw=2)

    ax.axvline(rf.P1, color='purple', linestyle=':', lw=2, alpha=0.8)
    
    if proc['lambda_tilde']:
        ax.axvline(proc['lambda_tilde'], color='orange', ls='-.', lw=1.5, alpha=0.6)

    ax.set_ylabel(r'$[x_3]$', rotation=0, ha='center', va='bottom')
    ax.yaxis.set_label_coords(-0.06, 0.96) 
    ax.set_xlabel(r'$\lambda$', ha='right', va='top')
    # ax.xaxis.set_label_coords(1.00, -0.05)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

    ax.set_title(r'Continuation Profile for $\lambda = p_1$', pad=30, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.subplots_adjust(right=0.82, top=0.85, bottom=0.15)
    plt.show()

def plot_stability_test(proc):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_with_stability(ax, proc['p'], proc['eps'], proc['re_h'], color='black', lw=2)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    
    if proc['lambda_tilde']:
        ax.plot(proc['lambda_tilde'], 0, 'o', ms=10, 
                markeredgecolor='orange', markerfacecolor='none', markeredgewidth=2, zorder=5)

    idx_default = np.argmin(np.abs(proc['p'] - rf.P1))
    eps_P1 = proc['eps'][idx_default]

    ax.plot(rf.P1, eps_P1, 'o', ms=10, color='purple')

    ax.set_ylabel(r'$\varepsilon(\lambda)$', rotation=0, ha='center', va='bottom')
    ax.yaxis.set_label_coords(-0.08, 0.95) 
    ax.set_xlabel(r'$\lambda$', ha='right', va='top')
    # ax.xaxis.set_label_coords(1.05, -0.05)

    ax.set_title('Stability Test', pad=30, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.subplots_adjust(right=0.82, top=0.85, bottom=0.15)
    plt.show()

def plot_frequency_profile(proc):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    plot_with_stability(ax, proc['p'], proc['im_hopf'], proc['re_h'], color='black', lw=2, label=r'$|Im(\mu_{1,2})|$')
    
    if proc['lambda_tilde']:
        idx_tilde = np.argmin(np.abs(proc['p'] - proc['lambda_tilde']))
        ax.plot(proc['lambda_tilde'], proc['im_hopf'][idx_tilde], 'o', mfc='none', mec='orange', mew=2, markersize=10)

    idx_default = np.argmin(np.abs(proc['p'] - rf.P1))
    eps_P1 = proc['im_hopf'][idx_default]

    ax.plot(rf.P1, eps_P1, 'o', ms=10, color='purple')
    ax.set_ylabel(r'$|\omega|$', rotation=0, labelpad=20, va='center')
    ax.yaxis.set_label_coords(-0.04, 1.0) 
    ax.set_xlabel(r'$\lambda$')
    ax.set_title('Frequency Plot')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lambda_zero = 0.0
    lambda_final = 1.0
    steps = 1000
    tol = 1e-4 

    raw_data = run_continuation(lambda_zero, lambda_final, steps)

    if raw_data:
        res = process_results(raw_data, tol)
        hopf_criteria = res['hopf_flags']
        existence, non_degeneracy, transversality = hopf_criteria
    
        #plot_continuation_profile(res)
        #plot_stability_test(res)
        plot_frequency_profile(res)
        
        # Hopf criteria check
        if all(hopf_criteria):
            l_tilde = res['lambda_tilde']
            idx = np.argmin(np.abs(res['p'] - l_tilde))
            omega = res['im_hopf'][idx]
            period_limit_cycle = (2*np.pi)/omega

            state_at_bif = res['x'][idx]
            x1 = state_at_bif[0]
            x2 = state_at_bif[1]
            x3 = state_at_bif[2]

            print(f"Hopf Bifurcation confirmed at lambda = {l_tilde:.2f}")
            print(f"State Space: ({x1:.2f}, {x2:.2f}, {x3:.2f})")
            print(f"Period of limit cycle: {period_limit_cycle:.2f}")

        else:
            print(f"Existence: {existence}") 
            print(f"Non-Degeneracy: {non_degeneracy}")
            print(f"Transversality: {transversality}")
            print(f"No Hopf bifurcation occured")