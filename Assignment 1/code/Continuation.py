import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import Discretization as disc
import RootFinding as rf

def compute_explicit_jacobian_local(a_coeffs, D):
    temp_nodes = disc.P @ a_coeffs
    solar_q = disc.SOLAR_RADIATION * (disc.P[:, 0] - disc.LEGENDRE_COEFFICIENT * disc.P[:, 2])
    
    dimless_temp = disc.INVERSE_TEMPERATURE * (temp_nodes - disc.MIXED_POINT)
    dtanh = disc.INVERSE_TEMPERATURE * (1 - np.tanh(dimless_temp)**2)
    dalbedo_dt = 0.5 * (disc.ALBEDO_WATER - disc.ALBEDO_ICE) * dtanh
    
    heat_fluxR = (-solar_q * dalbedo_dt) - (4 * disc.EMISSIVITY * disc.BOLTZMANN_CONSTANT * temp_nodes**3)
    H_res = (disc.P.T * (disc.weights * heat_fluxR)) @ disc.P
    
    m = np.arange(len(a_coeffs))
    ortho = 2 / (2 * m + 1)
    K = np.diag(D * m * (m + 1) * ortho)
    Jacobian = H_res - K

    return Jacobian

def get_residual_mu(a_coeffs, mu, D):
    temperature_nodes = disc.P @ a_coeffs
    solar_q = disc.SOLAR_RADIATION * (disc.P[:, 0] - disc.LEGENDRE_COEFFICIENT * disc.P[:, 2])
    
    albedo = disc.ALBEDO_ICE + 0.5 * (disc.ALBEDO_WATER - disc.ALBEDO_ICE) * (
        1 + np.tanh(disc.INVERSE_TEMPERATURE * (temperature_nodes - disc.MIXED_POINT))
    )
    
    solar_radiation = solar_q * (1 - albedo) + mu
    emission_radiation = disc.EMISSIVITY * disc.BOLTZMANN_CONSTANT * temperature_nodes**4
    nonlinear_residual = ((solar_radiation - emission_radiation) * disc.weights) @ disc.P
    
    m = np.arange(len(a_coeffs))
    ortho = 2 / (2 * m + 1)
    dispersion = D * m * (m + 1) * a_coeffs * ortho

    residual = nonlinear_residual - dispersion
    
    return residual

def get_df_dmu(n):
    J_mu = disc.weights @ disc.P
    return J_mu

def arc_step(a_j, mu_j, D, tangent, ds, zeta):
    a_curr = a_j + tangent[:len(a_j)] * ds
    mu_curr = mu_j + tangent[-1] * ds
    
    n = len(a_j)
    
    for i in range(1, 25):
        F = get_residual_mu(a_curr, mu_curr, D)
        N_res = zeta * np.sum((a_curr - a_j)**2) + (1-zeta) * (mu_curr - mu_j)**2 - ds**2
        R_aug = np.append(F, N_res)
        
        dF_da = compute_explicit_jacobian_local(a_curr, D) 
        dF_dmu = disc.weights @ disc.P 
        
        row_a = 2 * zeta * (a_curr - a_j)
        row_mu = 2 * (1 - zeta) * (mu_curr - mu_j)
        
        J_aug = np.zeros((n + 1, n + 1))
        J_aug[:n, :n] = dF_da
        J_aug[:n, n] = dF_dmu
        J_aug[n, :n] = row_a
        J_aug[n, n] = row_mu
        
        try:
            delta = np.linalg.solve(J_aug, -R_aug)
            a_curr += delta[:n]
            mu_curr += delta[n]
            
            if np.linalg.norm(delta) < 1e-8:
                return a_curr, mu_curr, i
            
        except np.linalg.LinAlgError:
            return None, None, i
            
    return None, None, 25

def normalize_tangent(delta_a, delta_mu, zeta):
    weighted_norm = np.sqrt(zeta * np.sum(delta_a**2) + (1 - zeta) * (delta_mu**2))
    normalized_tangent = np.append(delta_a, delta_mu) / weighted_norm
    return normalized_tangent

def run_continuation(a_start, mu_start, mu_final, D, ds_max, zeta):
    a = a_start.copy()
    mu = mu_start
    ds = ds_max
    
    # Initialize tangent
    tangent = np.append(np.zeros_like(a), 1.0)
    
    results = [np.concatenate([[mu], a])] 
    max_steps = 1000 # Increased for D=0.003
    step_count = 0
    
    while step_count < max_steps:
        a_new, mu_new, iters = arc_step(a, mu, D, tangent, ds, zeta)

        if a_new is None or mu_new is None:
            ds *= 0.5
            print(f"Step failed at mu ~ {mu:.4f}, shrinking ds to {ds:.6f}")

            if ds < 1e-7: # Tighter tolerance for low D
                print("Convergence lost: ds below threshold.")
                break
            continue 

        if mu_new < -100 or mu_new > 200: 
            print(f"Safety bound for mu breached: {mu_new:.2f}")
            break
        
        da = a_new - a
        dmu = mu_new - mu
        
        new_tangent = normalize_tangent(da, dmu, zeta)
        if np.dot(new_tangent, tangent) < 0:
            new_tangent = -new_tangent
        
        a, mu, tangent = a_new, mu_new, new_tangent
        
        I_NEWTON = 4 
        xi = np.clip(I_NEWTON / max(iters, 1), 0.5, 2.0)
        ds = np.clip(ds * xi, 1e-7, ds_max)
        
        results.append(np.concatenate([[mu], a]))

        # Termination condition
        if (mu_final > mu_start and mu >= mu_final) or (mu_final < mu_start and mu <= mu_final):
            print(f"Reached mu_final: {mu:.2f}")
            break 

        step_count += 1
    
    return np.array(results)

def find_bifurcation_points(results, D):
    leading_evs = get_stability_data(results, D) 
    
    sign_change = np.diff(np.sign(leading_evs)) != 0
    bif_indices = np.where(sign_change)[0]
    
    return bif_indices, leading_evs

def get_stability_data(results, D):
    mu_vals = results[:, 0]
    a_coeffs_matrix = results[:, 1:]
    leading_evs = []

    for i in range(len(mu_vals)):
        a_vals = a_coeffs_matrix[i, :]
        
        J = compute_explicit_jacobian_local(a_vals, D)
        
        evs = np.linalg.eigvals(J)
        leading_evs.append(np.max(np.real(evs)))
        
    return np.array(leading_evs)

def plot_climate_analysis(results):
    mu = results[:, 0]
    a0 = results[:, 1]
    
    # 1. Identify turning points
    max_idx = argrelextrema(mu, np.greater, order=5)[0]
    min_idx = argrelextrema(mu, np.less, order=5)[0]
    tp_indices = np.sort(np.concatenate([max_idx, min_idx]))
    splits = np.concatenate([[0], tp_indices, [len(mu)-1]])

    num_branches = len(tp_indices) + 1
    cmap = plt.get_cmap('gist_rainbow_r')
    colors = [cmap(i) for i in np.linspace(0, 1, num_branches)]

    branch_configs = {}
    for i in range(num_branches):
        branch_configs[i] = {
            'c': colors[i],
            'l': f'Stable' if i % 2 == 0 else f'Unstable' ,
            'ls': '-' if i % 2 == 0 else '--', 
            'lw': 2.0
        }

    # Figure 1: Bifurcation Diagram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(len(splits) - 1):
        start, end = splits[i], splits[i+1] + 1
        cfg = branch_configs.get(i)
        if cfg:
            ax1.plot(mu[start:end], a0[start:end], 
                     color=cfg['c'], ls=cfg['ls'], label=cfg['l'], lw=cfg['lw'])

    ax1.plot(mu[tp_indices], a0[tp_indices], 'ko', mfc='none', ms=8, label='Bifurcations')
    ax1.set_title(rf'Bifurcation Diagram ($D = {D}$)')
    ax1.set_xlabel(r"$\mu$ [W/m$^2$]", fontsize=12)
    ax1.set_ylabel(r'$a_0^*$ [K]', rotation=0, ha='right')
    ax1.yaxis.set_label_coords(-0.03, 0.95)
    ax1.set_ylim(np.min(a0) - 10, np.max(a0) + 20)
    ax1.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

    plt.tight_layout()

    # Figure 2: Temperature Profile
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(-1, 1, 200)
    target_mu = 30 
    
    idx_slice = np.where(np.abs(mu - target_mu) < 0.25)[0]
    if len(idx_slice) > 0:
        sorted_slice = idx_slice[np.argsort(a0[idx_slice])]
        for idx in sorted_slice:
            segment_idx = np.searchsorted(splits, idx) - 1
            segment_idx = max(0, segment_idx)
            cfg = branch_configs.get(segment_idx)
            
            if cfg:
                T_coeffs = results[idx, 1:]
                T_vals = np.polynomial.legendre.legval(x_grid, T_coeffs)
                label_str = rf"{cfg['l']}"
                ax2.plot(x_grid, T_vals, color=cfg['c'], ls=cfg['ls'], lw=cfg['lw'], label=label_str)

    ax2.axhline(273, color='gray', ls=':', alpha=0.6)
    ax2.set_title(r'Temperature Profile around $\mu = 30$ W/m$^2$')
    ax2.set_xlabel(r'$x = \sin \theta$')
    ax2.set_ylabel(r'$T(x)$ [K]', rotation=0, ha='right')
    ax2.yaxis.set_label_coords(-0.03, 1.00)
   
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig1, fig2

def plot_eigenvalue_spectrum(results, D):
    mu_vals = results[:, 0]
    a_coeffs = results[:, 1:]
    
    leading_evs = []
    
    for i in range(len(mu_vals)):
        J = compute_explicit_jacobian_local(a_coeffs[i, :], D)
        evs = np.linalg.eigvals(J)
        leading_evs.append(np.max(np.real(evs)))
    
    leading_evs = np.array(leading_evs)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.axhline(0, color='black', lw=1, ls='-')
    ax.plot(mu_vals, leading_evs, color='black', ls="--", lw=1.5)
    ax.fill_between(mu_vals, 0, leading_evs, where=(leading_evs > 0), color='red', alpha=0.3, label= rf'Unstable ($\lambda > 0$)')
    ax.fill_between(mu_vals, 0, leading_evs, where=(leading_evs < 0), color='green', alpha=0.1, label=rf'Stable ($\lambda < 0$)')

    zero_crossings = np.where(np.diff(np.sign(leading_evs)) != 0)[0]
    ax.plot(mu_vals[zero_crossings], leading_evs[zero_crossings], 'ko', mfc='yellow', ms=8)

    ax.set_title(r"Stability Spectrum: $\lambda_{max}$ vs $\mu$", fontsize=14)
    ax.set_xlabel(r"$\mu$ [W/m$^2$]", fontsize=12)
    ax.set_ylabel(r"Re($\lambda_{max}$)", rotation=0, ha="right", fontsize=12)
    ax.yaxis.set_label_coords(-0.03, 1.00)
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    #plt.tight_layout()
    #fig.savefig(r'figures/figure7.pdf', bbox_inches='tight')

    plt.show()

    return fig

if __name__ == "__main__":
    # Start with your guess
    a_guess = np.zeros(disc.NUMBER_OF_LEGENDRE_POLYNOMIALS + 1)
    a_guess[0] = 230
    mu_0 = 0
    mu_final = 100
    zeta = 0.5
    ds = 1
    D = disc.DISPERSION_COEFFICIENT                #(5c)
    #D = 30                                         #(5e) - easy
    #D = 0.003                                       #(5e) - hard

    # Pre-solve to settle 
    a_stable, iters, _, success = rf.newton_raphson(
        lambda a: get_residual_mu(a, mu_0, D), 
        a_guess, 
        jac_type='explicit', 
        damping=False
    )

    # Start continuation
    if success:
        print(f"True starting state found in {iters} iterations.")
        print(f"Stable a_coeffs: {a_stable}")
        results = run_continuation(a_stable, mu_0, mu_final, D, ds,  zeta)
        for n in range(1, len(results)):
            if results[n] is None:
                print(n)
        tp_indices, ev_history = find_bifurcation_points(results, D)

        bif_mu = results[tp_indices, 0]
        bif_a0 = results[tp_indices, 1]
        
        plot_climate_analysis(results)

        for idx in tp_indices:
            print(f"Bifurcation at mu={results[idx,0]:.2f}, Leading EV={ev_history[idx]:.2e}")

        plot_eigenvalue_spectrum(results, D)

    else:
        print("Could not find a valid starting equilibrium at mu=0.")