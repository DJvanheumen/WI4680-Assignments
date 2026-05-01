import numpy as np
import matplotlib.pyplot as plt
import Results as res 
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import least_squares


# Default parameter values
P2, P3, P4, P5, P6, P7, P8 = 2.5, 0.6, 1.5, 4.5, 1.0, 0.2, 0.5

def vector_field(y, p1):
    x1, x2, x3 = y

    dx1 = x1*(1-x1) - p1*x1*x2 - P2*x1*x3
    dx2 = P3*x2*(1-x2) - P4*x1*x2
    dx3 = (P5*x1*x3)/(x1+P6) - P7*x1*x3 - P8*x3

    return np.array([dx1, dx2, dx3])

def get_jac(y, p1):
    eps = 1e-7
    J = np.zeros((3, 3))
    f0 = vector_field(y, p1)

    for i in range(3):
        y_p = y.copy(); y_p[i] += eps
        J[:, i] = (vector_field(y_p, p1) - f0) / eps

    return J

def dynamics(t, state_aug, T, p1):
    y = state_aug[:3]
    Phi = state_aug[3:].reshape((3, 3))
    dydt = T * vector_field(y, p1)
    dPhidt = T * (get_jac(y, p1) @ Phi)

    return np.concatenate([dydt, dPhidt.flatten()])


def shoot(u, y_anchor):
    y0, T, p1 = u[:3], u[3], u[4]

    sol = solve_ivp(dynamics, [0, 1], np.concatenate([y0, np.eye(3).flatten()]), 
                    args=(T, p1), rtol=1e-9, atol=1e-11, method='Radau')
    
    yf, M = sol.y[:3, -1], sol.y[3:, -1].reshape((3, 3))
    
    v_anchor = vector_field(y_anchor, p1)
    R = np.append(yf - y0, np.dot(v_anchor, (y0 - y_anchor)))
    
    J_s = np.zeros((4, 5))
    J_s[:3, :3] = M - np.eye(3)
    J_s[:3, 3] = T * vector_field(yf, p1) 
    
    eps_p = 1e-8

    sol_p = solve_ivp(dynamics, [0, 1], np.concatenate([y0, np.eye(3).flatten()]), 
                      args=(T, p1 + eps_p), rtol=1e-9, atol=1e-11, method='Radau')
    
    R_p = np.append(sol_p.y[:3, -1] - y0, np.dot(v_anchor, (y0 - y_anchor)))
    J_s[:, 4] = (R_p - R) / eps_p

    return R, J_s, M


def run_continuation(u_init):
    u_curr = u_init
    y_anchor = u_init[:3].copy()

    # Increase the bias for T in the initial tangent
    tau = np.array([0, 0, 0, 1.0, 0.0]) 
    tau /= np.linalg.norm(tau)
    #ds = 0.1
    ds = 0.05 
    #ds = 0.005

    print(f"{'Step':<5} | {'p1':<8} | {'T':<8} | {'Acc (+1)':<10} | {'PDB (-1)':<10}")
    print("-" * 65)

    history = []

    for i in range(250):
        u_pred = u_curr + ds * tau
        u_next = u_pred.copy()
        
        success = False

        for _ in range(15):
            R, J, M = shoot(u_next, y_anchor)
            R_aug = np.append(R, np.dot(u_next - u_curr, tau) - ds)
            J_aug = np.vstack([J, tau])
            
            step, _, _, _ = np.linalg.lstsq(J_aug, -R_aug, rcond=None)
            u_next += np.clip(step, -0.05, 0.05)
            
            if np.linalg.norm(step) < 1e-8:
                success = True
                break
        
        if success:
            if u_next[3] < u_curr[3]:
                tau = -tau
                continue 

            mu = np.linalg.eigvals(M)
            acc_plus = np.min(np.abs(mu - 1.0))
            acc_minus = np.min(np.abs(mu + 1.0))

            history.append(np.concatenate([u_curr, [acc_plus, acc_minus]]))
            
            # Predict next tangent
            _, J_f, _ = shoot(u_next, y_anchor)
            _, _, Vh = np.linalg.svd(J_f)
            new_tau = Vh[-1, :]

            if np.dot(new_tau, tau) < 0: 
                new_tau = -new_tau
            
            tau, u_curr, y_anchor = new_tau, u_next, u_next[:3].copy()
            
            if i % 5 == 0:
                print(f"{i:<5} | {u_curr[4]:.5f} | {u_curr[3]:.3f} | {acc_plus:.1e} | {acc_minus:.1e}")
            
            if acc_minus < 0.05:
                print(f"!!! PDB NEAR: p1 = {u_curr[4]:.5f}, T = {u_curr[3]:.3f} !!!")

            #ds = min(ds * 1.2, 0.1)
            ds = min(ds * 1.25, 0.05)
            #ds = min(ds * 1.1, 0.05)

            if u_curr[3] > 150: 
                break 
        else:
            ds /= 2.0
            if ds < 1e-6: 
                break

    return np.array(history)

def perfect_orbit_objective(vars, p1, y_anchor):
    y0 = vars[:3]
    T = vars[3]
    
    sol = solve_ivp(dynamics, [0, 1], np.concatenate([y0, np.eye(3).flatten()]), 
                    args=(T, p1), rtol=1e-9, atol=1e-11, method='Radau')
    yf = sol.y[:3, -1]
    
    v_anchor = vector_field(y_anchor, p1)
    phase = np.dot(v_anchor, (y0 - y_anchor))
    
    orbit = np.append(yf - y0, phase)

    return orbit

def robust_refine(y_unsettled, T_guess, p1_val):
    print("Launching Robust Refiner (Least Squares)...")
    initial_guess = np.append(y_unsettled, T_guess)
    
    res = least_squares(perfect_orbit_objective, initial_guess, 
                        args=(p1_val, y_unsettled),
                        ftol=1e-10, xtol=1e-10, gtol=1e-10,
                        method='lm') 
    
    if res.success:
        print(f"Refinement Successful! Final Cost: {res.cost:.2e}")
        return res.x # 
    
    else:
        print("Refinement failed to converge.")
        return None

if __name__ == "__main__":
    # Test at a point where you expect a cycle
    # test_p1 = 0.75 
    # y0 = [0.15, 0.65, 0.14] # Perturbed from fixed point

    # sol = solve_ivp(lambda t, y: vector_field(y, test_p1), [0, 200], y0, rtol=1e-9)

    # Integrate and track the peaks of Tumor (x1)
    # sol = solve_ivp(lambda t, y: vector_field(y, 0.75), [0, 1000], [0.15, 0.65, 0.14], 
    #             rtol=1e-10, atol=1e-12, method='Radau')


    # peaks, _ = find_peaks(sol.y[0])

    # # Look at the last few peak values
    # print("Last 5 peak values of x1:")
    # print(sol.y[0, peaks[-5:]])

    # y_start = sol.y[:, 0] 

    # # 2. End point (The Seed on the Limit Cycle)
    # y_seed = sol.y[:, -1] 

    # plt.figure(figsize=(10, 6))
    # plt.plot(sol.y[0], sol.y[2], 'b-', alpha=0.5, label='Trajectory')

    # # Mark the points
    # plt.scatter(y_start[0], y_start[2], color='green', s=100, label='Start (Kick)', zorder=5)
    # plt.scatter(y_seed[0], y_seed[2], color='red', s=100, label='End (Limit Cycle Seed)', zorder=5)

    # plt.xlabel('Tumor (x1)')
    # plt.ylabel('Immune (x3)')
    # plt.legend()
    # plt.title(f'Trajectory Start vs. End at p1 = 0.75')
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # print(f"Start Point: {y_start}")
    # print(f"Final Seed for Solver: {y_seed}")

    # Refinement
    y_guess = res.continuation_data[-1][0:3]
    T_guess = res.continuation_data[-1][3]
    p1_guess = res.continuation_data[-1][4]

    u_refined = robust_refine(y_guess, T_guess, p1_guess)
    np.save('Refinement.npy', u_refined)
    print(u_refined)
   
    # if u_refined is not None:
    #     u_init = np.append(u_refined, p1_start)
    #     run_continuation(u_init)

    # Old seed (before continuation, after refinement)
    # y_seed = [0.09074417, 0.80924822, 0.0524495] 
    # T_seed = 19.42613129 
    # p1_seed = 0.75

    # New seed (after 1 continuation and a time integration step)
    # y_seed = [0.09269057, 0.82333397,  0.02299244]
    # T_seed = 21.70334301
    # p1_seed = 0.78784

    # New seed (after 2 continuations and refinement)
    # y_seed = [9.33583103e-02, 8.25081901e-01, 1.12108918e-02]
    # T_seed = 2.41956832e+01
    # p1_seed = 0.82647

    # New seed (after 3 continuations and refinement)
    # y_seed = [9.32258213e-02, 8.25086926e-01, 9.56116540e-03]
    # T_seed = 2.48078143e+01
    # p1_seed = 0.83551 

    # New seed (after 4 continuations and refinement)
    # y_seed = [9.26727789e-02, 8.24891137e-01,6.83770265e-03]
    # T_seed = 2.61680567e+01
    # p1_seed = 0.85496

    # New seed (after 5 continuations and refinement)
    # y_seed = res.refinement_data[0:3]
    # T_seed = res.refinement_data[-1]
    # p1_seed = 0.87434


    # New seed (after 6 continuations and refinement)
    # y_seed = res.refinement_data[0:3]
    # T_seed = res.refinement_data[-1]
    # p1_seed = res.continuation_data[-1][4]
    
    # u_init = np.array([*y_seed, T_seed, p1_seed])

    # print(f"Starting continuation from p1 = {p1_seed}, T = {T_seed:.3f}...")

    # u_curr = run_continuation(u_init)
    # np.save('PContinuation.npy', u_curr)