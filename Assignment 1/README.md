# Assignment 1: The Earth's Energy Balance

Contains code and figures corresponding to the first assignment about the effects of climate change 
on the temperature distribution in a simplified one dimensional space. 

## Code

### Requirements

Install python and pip. Use pip to install the python libraries: numpy, scipy and matplotlib

### Discretization.py

Based on default parameters this program calculates a Legendre Polynomial basis P based on Gaussian Quadratures for a number of nodes N. Then it calculate the total residual based on explicit terms for diffusion, solar absorption and black body emission coming from physical models using the Legendre basis P. 

In order to calculate the total residual: supply a guess vector a with the length equal to N + 1, the first coefficient a0 and then call the function. 

### RootFinding.py

Using the calculated total residual obtained from Discretization.py this program finds roots using the Newton-Raphson and Broyden method for an initial guess range. First a Jacobian J(a) is calculated, here two methods are used: an explicit analytical derivation and a numerical approximation using finite differences with a step size h. 

The number of iterations depends of the value of the dispersion coefficient D. For D = 0.3, Newton-Raphson converges under 10 iterations and Broyden under 20 iterations. After the convergence range is sweeped, an example plot is provided. 

### PerformanceTest.py

An example program to test performance between explicit and finite difference. The explicit method is typically more accurate and around 2 times faster.

### Continuation.py

A standard pseudo-arc length continuation routine in the parameter mu is implementated here. For the predictor Euler Forwards is used and for the corrector Newton-Raphson is used with a step size ds. A maximal and minimal stepsize are needed for a smooth continuation. Bifurcation points are also calculated based on continuation results along with their stability in a small neighborhood around them. An example plot is provided.

# Figures

Example figures are provided using the Python code above, the current code may need adapted to suit the need of the user.

# Report 

The related report is found at:

https://www.overleaf.com/project/69cbe2ab74c4eccffeae8f70
