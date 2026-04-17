import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Discretization as disc

TOLERANCE = 1e-9
MAX_NUMBER_OF_ITERATIONS = 50       # Newton
MAX_NUMBER_OF_ITERATIONS_B = 50     # Broyden
ALPHA = 1e-4
D = disc.DISPERSION_COEFFICIENT     # Normal
#D = 30                              # Easy
#D = 0.003                            # Hard   

def compute_explicit_jacobian(a_coeffs):
    temp_nodes = disc.P @ a_coeffs
    
    dimless_temp = disc.INVERSE_TEMPERATURE * (temp_nodes - disc.MIXED_POINT)
    dtanh = disc.INVERSE_TEMPERATURE * (1 - np.tanh(dimless_temp)**2)
    dalbedo_dt = 0.5 * (disc.ALBEDO_WATER - disc.ALBEDO_ICE) * dtanh
    solar_q = disc.SOLAR_RADIATION * (disc.P[:, 0] - disc.LEGENDRE_COEFFICIENT * disc.P[:, 2])
    
    # dRA/dalpha, dRE/dalpha
    heat_fluxRA = -solar_q * dalbedo_dt
    heat_fluxRE = 4 * disc.EMISSIVITY * disc.BOLTZMANN_CONSTANT * temp_nodes**3
    
    heat_fluxR = heat_fluxRA - heat_fluxRE
    
   
    H_res = (disc.P.T * (disc.weights * heat_fluxR)) @ disc.P
    
    # Diffusion part K
    m = np.arange(len(a_coeffs))
    ortho = 2 / (2 * m + 1)
    K_diag = D * m * (m + 1) * ortho
    
    return H_res - np.diag(K_diag)

def compute_jacobian(f, a, h=1e-5):
    N = len(a)
    J = np.zeros((N, N))
    for k in range(N):
        ah_plus = a.copy()
        ah_minus = a.copy()
        ah_plus[k] += h
        ah_minus[k] -= h
        J[:, k] = (f(ah_plus) - f(ah_minus)) / (2 * h)
    return J

def newton_raphson(f, a0, jac_type, damping):
    a = a0.copy()
    gamma_min = 1.0
    
    for n in range(MAX_NUMBER_OF_ITERATIONS):
        res = f(a)
        res_norm = np.linalg.norm(res)
        
        if res_norm < TOLERANCE:
            if 150 < a[0] < 450:
                return a, n, gamma_min, True
            else:
                return a, n, gamma_min, False
            
        if jac_type == 'explicit':
            J = compute_explicit_jacobian(a)

        elif jac_type == 'fd':
            J = compute_jacobian(f, a) 

        else:
            print('This method is currently not supported')
            return a, n, 0.0, False
        
        try:
            delta_a = np.linalg.solve(J, -res)

            if damping:
                gamma = 1.0

                while gamma > ALPHA:
                    a_new = a + gamma * delta_a

                    if a_new[0] < 150 or a_new[0] > 400:
                        gamma *= 0.5
                        continue
                        
                    res_next_norm = np.linalg.norm(f(a_new))

                    if res_next_norm < (1 - ALPHA * gamma) * res_norm:
                        break

                    gamma *= 0.5 

                a = a_new
                a[1::2] = 0.0
                gamma_min = min(gamma, gamma_min)

            else:
                a += delta_a

        except np.linalg.LinAlgError:
            return a, n, 0.0, False
            
    return a, MAX_NUMBER_OF_ITERATIONS, gamma_min, False

def broyden(f, a0, jac_type):
    a = a0.copy()
    f0 = f(a)

    if jac_type == 'explicit':
        B = compute_explicit_jacobian(a)

    elif jac_type == 'fd':
        B = compute_jacobian(f, a)

    else:
        print("This method is currently not supported")
        return a, 0.0, False

    for n in range(MAX_NUMBER_OF_ITERATIONS_B):
        try:
            delta_a = np.linalg.solve(B, -f0)
        except np.linalg.LinAlgError:
            return a, n, False
        
        a_new = a + delta_a
        f_new = f(a_new)
        
        if np.linalg.norm(f_new) < TOLERANCE:
            return a_new, n + 1, True
        
        norm_delta_sq = np.dot(delta_a, delta_a)
        
        B = B + np.outer(f_new, delta_a) / norm_delta_sq
        
        a = a_new
        f0 = f_new
        
    return a, MAX_NUMBER_OF_ITERATIONS_B, False

def analyze_convergence_range(temp_min, temp_max, steps, jac_type, method, damping):
    initial_temps = np.linspace(temp_min, temp_max, steps)
    results = []
    jacobian_errors = [] 
    
    print(f"Sweeping {method} Jacobian from {temp_min} K to {temp_max} K...")
    
    for T_start in initial_temps:
        a_guess = np.zeros(disc.NUMBER_OF_LEGENDRE_POLYNOMIALS + 1)
        a_guess[0] = T_start
        
        if method == 'performance':
            j_exp = compute_explicit_jacobian(a_guess)
            j_fd = compute_jacobian(disc.calculate_residual, a_guess, h=1e-5)
            err = np.linalg.norm(j_exp - j_fd, ord='fro')
            jacobian_errors.append(err)

        elif method == 'broyden':
            sol, iters, success = broyden(
                disc.calculate_residual,
                a_guess,
                jac_type=jac_type
            )

            if success:
                results.append((T_start, iters, sol[0], 1.0)) 

            else:
                results.append((T_start, -1, np.nan, 0.0)) 
        
        elif method == 'newton':
            sol, iters, gamma, success = newton_raphson(
                disc.calculate_residual, 
                a_guess, 
                jac_type=jac_type,
                damping=damping
            )
            
            if success:
                results.append((T_start, iters, sol[0], gamma)) 

            else:
                results.append((T_start, -1, np.nan, 0.0)) 
                
    return np.array(results), np.array(jacobian_errors)

def get_valid(data):
        mask = data[:, 1] != -1
        return data[mask, 0], data[mask, 1]


if __name__ == "__main__":
    temp_min = 220
    temp_max = 320
    steps = int(temp_max - temp_min)
    num_points = steps + 1 
    
    data_newton, _ = analyze_convergence_range(temp_min, temp_max, num_points, 'explicit', 'newton', damping=True)
    data_broyden, _ = analyze_convergence_range(temp_min, temp_max, num_points, 'explicit', 'broyden', damping=False)

    x_newton, i_newton = get_valid(data_newton)
    x_broyden, i_broyden = get_valid(data_broyden)

    mask_n = data_newton[:, 1] != -1
    a0_star_newton = data_newton[mask_n, 2]
    x_star_newton = data_newton[mask_n, 0]

    mask_b = data_broyden[:, 1] != -1
    a0_star_broyden = data_broyden[mask_b, 2]
    x_star_broyden = data_broyden[mask_b, 0]

    print(f"Newton Successes: {np.sum(mask_n)} / {num_points}")
    print(f"Broyden Successes: {np.sum(mask_b)} / {num_points}")

    newton_fails = data_newton[~mask_n, 0]
    print(f"Newton failed at: {newton_fails}")

    broyden_fails = data_broyden[~mask_b, 0]
    print(f"Broyden failed at: {broyden_fails}")

    unique_newton = np.unique(np.round(a0_star_newton, decimals=6))
    unique_broyden = np.unique(np.round(a0_star_broyden, decimals=6))

    unique_points = np.intersect1d(unique_newton, unique_broyden)
    print(unique_points)
    
    print(f"Newton Unique States: {unique_newton}")
    print(f"Broyden Unique States: {unique_broyden}")
    
    if len(unique_newton) == len(unique_broyden):
        diff = np.abs(unique_newton - unique_broyden)

        if np.min(diff) < 1e-6:
            print(f"Looks like an unique point")
            print(diff)

        print(f"Difference: {np.max(diff):.9e} K")


    plt.rcParams.update({
        'font.size': 12,           
        'axes.titlesize': 16,      
        'axes.labelsize': 14,      
        'xtick.labelsize': 12,     
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'text.usetex': False       
    })

    # Figure 3: Convergence Point Comparison
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.scatter(x_star_newton, a0_star_newton, color='#1f77b4', label='Newton', alpha=0.4, s=50, marker='o')
    ax1.scatter(x_star_broyden, a0_star_broyden, color='#d62728', label='Broyden', alpha=0.9, s=30, marker='x')

    ax1.set_ylabel(r"$a_0^*$" + "[K]", rotation=0, ha='center', va='center')
    ax1.yaxis.set_label_coords(-0.06, 1.02) 
    ax1.set_xlabel(r"$a_0$" + " [K]")
    ax1.set_title("Climate Equilibria: Newton vs. Broyden Comparison")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='center right')
    # plt.savefig("\figures\figure8.pdf")
    plt.show()

    # Figure 4: Convergence Speed
    plt.figure(figsize=(10, 6))
    ax2 = plt.gca()
    ax2.plot(x_newton, i_newton, 'o-', color='#1f77b4', markersize=4, label='Newton', alpha=0.7)
    ax2.plot(x_broyden, i_broyden, 'o--', color='#d62728', markersize=4, label='Broyden', alpha=0.7)
    
    ax2.set_ylabel(r"$I$", rotation=0, ha='center', va='center')
    ax2.yaxis.set_label_coords(-0.04, 1.00) 
    ax2.set_xlabel(r"$a_0$" + " [K]")
    ax2.set_title("Convergence speed: Newton vs Broyden")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, loc='upper right')

    tick_spacing = 5
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    max_it_val = max(np.max(i_newton), np.max(i_broyden))
    upper_limit = (np.ceil(max_it_val / 5) + 1) * 5
    ax2.set_ylim(0, upper_limit-1)

    # plt.savefig("\figures\figure9.pdf", bbox_inches='tight')
    plt.show()