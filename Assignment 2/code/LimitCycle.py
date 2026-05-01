import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Parameters as defined in your source
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
    
    # Calculate initial tangent and force positive direction for p1
    _, J_init, _ = shoot(u_init, y_anchor)
    _, _, Vh = np.linalg.svd(J_init)
    tau = Vh[-1, :]
    
    # CRITICAL: Ensure we move in POSITIVE p1 direction (tau[4])
    if tau[4] < 0:
        tau = -tau
        
    ds = 0.005 # Small step to catch early bifurcations
    history = []

    print(f"{'Step':<5} | {'p1':<8} | {'T':<8} | {'PDB (-1)':<10}")
    
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
            mu = np.linalg.eigvals(M)
            acc_minus = np.min(np.abs(mu + 1.0)) # Tracking distance to PDB
            history.append(np.concatenate([u_curr, [acc_minus]]))
            
            _, J_f, _ = shoot(u_next, y_anchor)
            _, _, Vh = np.linalg.svd(J_f)
            new_tau = Vh[-1, :]
            if np.dot(new_tau, tau) < 0: new_tau = -new_tau
            
            tau, u_curr, y_anchor = new_tau, u_next, u_next[:3].copy()
            if i % 10 == 0:
                print(f"{i:<5} | {u_curr[4]:.5f} | {u_curr[3]:.3f} | {acc_minus:.1e}")
        else:
            break

    return np.array(history)

def perfect_orbit_objective(vars, p1, y_anchor):
    y0, T = vars[:3], vars[3]
    sol = solve_ivp(dynamics, [0, 1], np.concatenate([y0, np.eye(3).flatten()]), 
                    args=(T, p1), rtol=1e-9, atol=1e-11, method='Radau')
    yf = sol.y[:3, -1]
    v_anchor = vector_field(y_anchor, p1)

    return np.append(yf - y0, np.dot(v_anchor, (y0 - y_anchor)))

def robust_refine(y_guess, T_guess, p1_val):
    res = least_squares(perfect_orbit_objective, np.append(y_guess, T_guess), 
                        args=(p1_val, y_guess), ftol=1e-10, xtol=1e-10, method='lm')
    
    return res.x if res.success else None

if __name__ == "__main__":
    # Start near Hopf point found in previous sections
    p1_start = 0.71673
    x_init = [0.13, 0.67, 0.16]
    T_init = 17.56

    # 1. Pre-integrate to find the cycle
    sol = solve_ivp(lambda t, y: vector_field(y, p1_start), [0, 100], x_init, rtol=1e-9)
    
    # 2. Refine the seed
    refined = robust_refine(sol.y[:, -1], T_init, p1_start)
    
    if refined is not None:
        u_start = np.append(refined, p1_start)
        data = run_continuation(u_start)
        np.save('LimitCycle_Data.npy', data)