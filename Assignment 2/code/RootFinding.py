import numpy as np

# Parameters with default parameters
P1 = 0.5  
P2 = 2.5
P3 = 0.6
P4 = 1.5
P5 = 4.5
P6 = 1.0
P7 = 0.2
P8 = 0.5

def get_coexistence_state(p1):
    alpha = (P6 * P7 + P8 - P5)
    beta = P6 * P7 * P8
    
    # Using the biologically feasible minus sign as corrected
    discriminant = alpha**2 - 4*beta
    if discriminant < 0:
        print('Unfeasible solution')
        print(f'Discriminant less than zero): {discriminant}')
        return np.nan, np.nan, np.nan
        
    x1 = (-alpha - np.sqrt(discriminant)) / (2 * P7)
    x2 = 1 - (P4 / P3) * x1
    x3 = (1 - x1 - p1 * x2) / P2

    if x1 < 0 or x2 < 0 or x3 < 0:
        print(f'Unfeasable solution: ({x1:.2f}, {x2:.2f}, {x3:.2f}')
        return np.nan, np.nan, np.nan


    return x1, x2, x3

def calculate_jacobian(x1, x2, x3, p1):
    J = np.zeros((3, 3))
    J[0,:] = [1 - 2*x1 -1*p1*x2 -1*P2*x3, -1*p1*x1, -1*P2*x1]
    J[1,:] = [-1*P4*x2, P3*(1 - 2*x2) - 1*P4*x1, 0]
    J[2,:] = [(P5*P6*x3)/(x1 + P6)**2 - 1*P7*x3, 0, (P5*x1)/(x1 + P6) - 1*P7*x1 - 1*P8]
    return J

def compute_eigenvalues(J):
    eigenvalues, _ = np.linalg.eig(J)
    return eigenvalues

def analyze_stability(eigenvalues):
    real_parts = np.real(eigenvalues)
    max_real = np.max(real_parts)
    
    if max_real < 0:
        return "Stable equilibrium"
    elif max_real > 1e-9:
        return "Unstable equilibrium"
    else:
        return "Stable/Unstable (need further study)"

if __name__ == '__main__':
    
    p1 = P1

    # 1. Zero state
    x1 = 0
    x2 = 0
    x3 = 0
    jacobian0 = calculate_jacobian(x1, x2, x3, p1)
    lambda0 = compute_eigenvalues(jacobian0)
    results = analyze_stability(lambda0)
    print(f'Zero state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    #print(f'Jacobian ({x1}, {x2}, {x3}) : \n {jacobian0}')
    #print(f'Eigenvalues: {lambda0}')
    print(results)

    # 2. Tumor cells only state
    x1 = 1
    x2 = 0
    x3 = 0
    jacobianT = calculate_jacobian(x1, x2, x3, p1)
    lambdaT = compute_eigenvalues(jacobianT)
    results = analyze_stability(lambdaT)
    print(f'Tumor state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    print(f'Jacobian ({x1}, {x2}, {x3}) : \n {jacobianT}')
    print(f'Eigenvalues: {lambdaT}')
    print(results)

    # 3. Healthy tissue cells state
    x1 = 0
    x2 = 1
    x3 = 0
    jacobianH = calculate_jacobian(x1, x2, x3, p1)
    lambdaH = compute_eigenvalues(jacobianH)
    results = analyze_stability(lambdaH)
    print(f'Healthy state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    print(f'Jacobian ({x1}, {x2}, {x3}) : \n {jacobianH}')
    print(f'Eigenvalues: {lambdaH}')
    print(results)

    # 4. Tumor and healthy tissue cells
    x1 = (1 - P1)/(1 - P1*P4/P3)
    x2 = 1 - P4/P3*x1
    x3 = 0
    jacobianTH = calculate_jacobian(x1, x2, x3, p1)
    lambdaTH = compute_eigenvalues(jacobianTH)
    results = analyze_stability(lambdaTH)
    print(f'Tumor and Healthy competition state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    print(f'Jacobian ({x1}, {x2}, {x3}) : \n {jacobianTH}')
    print(f'Eigenvalues: {lambdaTH}')
    print(results)

    # 5. Coexistence states
    alpha = (P6*P7 + P8 - P5)
    beta = P6*P7*P8
    x1plus = -alpha/(2*P7) + np.sqrt(alpha**2 - 4*beta)/(2*P7)
    x1min = -alpha/(2*P7)  - np.sqrt(alpha**2 - 4*beta)/(2*P7)
    x1 = min(x1min, x1plus)
    x2 = 1 - P4/P3*x1
    x3 = (1 - x1 - P1*x2)/P2 
    jacobianTHE = calculate_jacobian(x1, x2, x3, p1)
    lambdaTHE = compute_eigenvalues(jacobianTHE)
    results = analyze_stability(lambdaTHE)
    print(f'Coexistence state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    print(f'Jacobian ({x1}, {x2}, {x3}): \n {jacobianTHE}')
    print(f'Eigenvalues: {lambdaTHE}')
    print(results)

    x1 = max(x1min, x1plus)
    x2 = 1 - P4/P3*x1
    x3 = (1 - x1 - P1*x2)/P2
    jacobianTHE = calculate_jacobian(x1, x2, x3, p1)
    lambdaTHE = compute_eigenvalues(jacobianTHE)
    results = analyze_stability(lambdaTHE)
    print(f'Coexistence state: ({x1:.2f}, {x2:.2f}, {x3:.2f})')
    print(f'Jacobian ({x1}, {x2}, {x3}): \n {jacobianTHE}')
    print(f'Eigenvalues: {lambdaTHE}')
    print(results)