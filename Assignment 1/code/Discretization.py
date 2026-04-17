import numpy as np
from scipy.special import roots_legendre

# parameters
SOLAR_RADIATION = 341.3                                 # Q_0 [Wm^-2]
ALBEDO_ICE = 0.7                                        # alpha_1
ALBEDO_WATER = 0.289                                    # alpha_2
EMISSIVITY = 0.61                                       # epsilon_0
BOLTZMANN_CONSTANT = 5.67*1e-8                          # sigma_0 [Wm^-2K^-4]
INVERSE_TEMPERATURE = 0.1                               # M [K^-1]
DISPERSION_COEFFICIENT = 0.3                            # D [Wm^-2K^-1]
#GREENHOUSE_EFFECT = 30                                 # mu [Wm^-2]
GREENHOUSE_EFFECT = 0                                   # mu [Wm^-2]
HEAT_CAPACITY = 5*1e8                                   # C_T [Jm^-2K^-1]
MIXED_POINT = 273                                       # T^* [K]
LEGENDRE_COEFFICIENT = 0.482                            # Order 2
NUMBER_OF_LEGENDRE_POLYNOMIALS = 6                     # N 
#NUMBER_OF_LEGENDRE_POLYNOMIALS = 24                     # N 

#D = DISPERSION_COEFFICIENT
#D = 30
D = 0.003

latitude_nodes, weights = roots_legendre(NUMBER_OF_LEGENDRE_POLYNOMIALS + 1)

def get_legendre_basis(n_max, x):
    P = np.zeros((len(x), n_max + 1))
    P[:, 0] = 1.0  
    if n_max > 0:
        P[:, 1] = x 
    for i in range(1, n_max):
        P[:, i+1] = ((2*i + 1) * x * P[:, i] - i * P[:, i-1]) / (i + 1)
    return P

# Pre-calculate the Transformation Matrix P
P = get_legendre_basis(NUMBER_OF_LEGENDRE_POLYNOMIALS, latitude_nodes)

def calculate_residual(a_coeffs):
    
    # T(x):= sum_n a_n*P_n(x)
    temperature_nodes = P @ a_coeffs
    
    # Q(x)
    solar_q = SOLAR_RADIATION * (P[:, 0] - LEGENDRE_COEFFICIENT * P[:, 2])
    
    # alpha(T(x))
    albedo = ALBEDO_ICE + 0.5 * (ALBEDO_WATER - ALBEDO_ICE) * (
        1 + np.tanh(INVERSE_TEMPERATURE * (temperature_nodes - MIXED_POINT))
    )
    
    # R_A(Q, alpha, mu), R_E(epsilon, sigma, T)
    solar_radiation = solar_q * (1 - albedo) + GREENHOUSE_EFFECT
    emission_radiation = EMISSIVITY * BOLTZMANN_CONSTANT * temperature_nodes**4
    
    heat_energy = solar_radiation - emission_radiation
    
    nonlinear_residual = (heat_energy * weights) @ P
    
    m = np.arange(len(a_coeffs))
    orthogonality_factor = 2 / (2 * m + 1)
    dispersion_projection = D * m * (m + 1) * a_coeffs * orthogonality_factor
    
    total_residual = nonlinear_residual - dispersion_projection
    return total_residual

# guess
# a = np.zeros(NUMBER_OF_LEGENDRE_POLYNOMIALS + 1)
# a[0] = 270
# results = calculate_residual(a)
# print(results)