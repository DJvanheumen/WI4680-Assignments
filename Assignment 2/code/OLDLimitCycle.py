import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Values from your Step 240
p1_val = 0.85496
T_val = 26.168
# Estimated state based on continuation trend
y_start = [0.0926,  0.8248, 0.0068]

def tumor_immune_system(t, y, p1):
    x1, x2, x3 = y
    # Fixed parameters from your earlier setup
    p2, p3, p4, p5, p6, p7, p8 = 0.1, 0.1, 0.1, 4.5, 1.0, 0.01, 0.5
    
    dx1 = x1*(1 - x1) - p1*x1*x2 - p2*x1*x3
    dx2 = p3*x2*(1 - x2) - p4*x1*x2
    dx3 = (p5*x1*x3)/(x1 + p6) - p7*x1*x3 - p8*x3
    return [dx1, dx2, dx3]

# Integrate for two periods
sol = solve_ivp(tumor_immune_system, [0, T_val * 2], y_start, 
                args=(p1_val,), method='Radau', rtol=1e-10, atol=1e-12)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 3D-like Projection
ax1.plot(sol.y[0], sol.y[2], color='crimson', lw=1.5)
ax1.set_xlabel('Tumor Cells (x1)')
ax1.set_ylabel('Effector Cells (x3)')
ax1.set_title(f'Limit Cycle at p1 = {p1_val:.4f}')
ax1.grid(alpha=0.3)

# Time Series
ax2.plot(sol.t, sol.y[0], label='Tumor (x1)', color='black')
ax2.plot(sol.t, sol.y[2], label='Immune (x3)', color='dodgerblue')
ax2.set_xlabel('Time (tau)')
ax2.set_ylabel('Population')
ax2.set_title(f'Temporal Evolution (T = {T_val:.2f})')
ax2.legend()

plt.tight_layout()
plt.show()