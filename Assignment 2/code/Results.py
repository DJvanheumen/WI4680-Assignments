import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def tumor_immune_system(t, y, p1):
    x1, x2, x3 = y
    # Fixed parameters from your earlier setup
    p2, p3, p4, p5, p6, p7, p8 = 0.1, 0.1, 0.1, 4.5, 1.0, 0.01, 0.5
    
    dx1 = x1*(1 - x1) - p1*x1*x2 - p2*x1*x3
    dx2 = p3*x2*(1 - x2) - p4*x1*x2
    dx3 = (p5*x1*x3)/(x1 + p6) - p7*x1*x3 - p8*x3
    return [dx1, dx2, dx3]

limitcycle_data = np.load('LimitCycle_Data.npy',allow_pickle=True)
print(limitcycle_data)

refinement_data = np.load('Refinement.npy', allow_pickle=True)
continuation_data = np.load('PContinuation.npy', allow_pickle=True)

if __name__ == "__main__":

    if continuation_data is not None:
        print(f"Array length: {len(continuation_data)}")
        print(continuation_data[-1])

        x1 = continuation_data[:, 0]
        x2 = continuation_data[:, 1]
        x3 = continuation_data[:, 2]
        Tc = continuation_data[:, 3]
        p1 = continuation_data[:, 4]
        acc = continuation_data[:, 5]
        floq = continuation_data[:,-1]

        plt.figure(figsize=(8, 6))
        plt.plot(x1, x3, 'b-', alpha=0.6)
        plt.xlabel('Tumor Cells ($x_1$)')
        plt.ylabel('Immune Effectors ($x_3$)')  
        plt.title('Phase Space Evolution during Continuation')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(p1, x3, 'b-', alpha=0.6)
        plt.xlabel(r'($\lambda$)')
        plt.ylabel(r'$[x_3]$)')  
        plt.title('Continuation diagram')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(p1, Tc, 'b-', alpha=0.6)
        plt.xlabel(r'($\lambda$)')
        plt.ylabel(r'$T_c$)')  
        plt.title('Continuation diagram')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        if refinement_data is not None:
            print(f"Array length: {len(refinement_data)}")
            print(refinement_data)

            y_start = [x1[0], x2[0], x3[0]]
            p1_start = p1[0]
            T_start = Tc[0]

            # Integrate for two periods
            sol = solve_ivp(tumor_immune_system, [0, 3*T_start], y_start, 
                            args=(p1_start,), method='Radau', rtol=1e-10, atol=1e-12)

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 3D-like Projection
            ax1.plot(sol.y[0], sol.y[2], color='crimson', lw=1.5)
            ax1.set_xlabel('Tumor Cells (x1)')
            ax1.set_ylabel('Effector Cells (x3)')
            ax1.set_title(f'Limit Cycle at p1 = {p1_start:.4f}')
            ax1.grid(alpha=0.3)

            # Time Series
            ax2.plot(sol.t, sol.y[0], label='Tumor (x1)', color='black')
            ax2.plot(sol.t, sol.y[2], label='Immune (x3)', color='dodgerblue')
            ax2.set_xlabel('Time (tau)')
            ax2.set_ylabel('Population')
            ax2.set_title(f'Temporal Evolution (T = {T_start:.2f})')
            ax2.legend()

            plt.tight_layout()
            plt.show()

