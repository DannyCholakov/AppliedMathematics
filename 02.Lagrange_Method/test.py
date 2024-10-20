import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize_scalar

# Define the symbolic variable and the function for approximation
t = sp.symbols('t')
f = t  # Function f(t) = t

# Define the point where we want to approximate the value
u = 1.15
xx = np.array([1.0, 1.1, 1.3])  # Given x values
n = len(xx)  # Number of points

# Evaluate f at given x values
yy = np.array([f.subs(t, x) for x in xx])

# Create pairs of (xx, yy)
sp_pairs = list(zip(xx, yy))

# Create a scatter plot of the points
plt.scatter(xx, yy, color='blue', label='Data Points')
plt.plot(xx, yy, color='red')  # Connect the data points with a red line
plt.title('Data Points for Interpolation')
plt.xlabel('xx values')
plt.ylabel('yy values')
plt.axhline(0, color='black', lw=0.5, ls='--')  # Add horizontal line at y=0
plt.axvline(0, color='black', lw=0.5, ls='--')  # Add vertical line at x=0
plt.grid()
plt.legend()
plt.show()

# Print xx and yy in a tabular format
print("xx values:", xx)
print("yy values:", yy)

# Define the Lagrange interpolation polynomial
L = sum(
    yy[i] * sp.prod((t - xx[j]) / (xx[i] - xx[j]) for j in range(n) if i != j)
    for i in range(n)
)

# Display the polynomial in a readable format
print(f"L(t) = {sp.expand(L)}")  # Display the polynomial

# Print the approximation at u
approximation_at_u = L.evalf(subs={t: u})
exact_value_at_u = f.subs(t, u)
print(f"L({u}) = {approximation_at_u:.5f}, точна стойност = {exact_value_at_u:.5f}")

# Calculate the n-th derivative of f(t)
n_derivative = sp.diff(f, t, n)  # n-th derivative of f(t)

# Convert the n-th derivative to a callable function
n_derivative_func = sp.lambdify(t, n_derivative, 'numpy')

# Find the maximum value of the absolute value of the n-th derivative
def objective_function(x):
    return -abs(n_derivative_func(x))  # Negate for minimization

result = minimize_scalar(objective_function, bounds=(xx[0], xx[-1]), method='bounded')
M = -result.fun  # Get the maximum value

print(f"Mn+1 = {M:.5f}")

# Calculate the remainder R
R = (M / sp.factorial(n)) * np.prod([abs(u - xx[i]) for i in range(n)])  # R = M / n! * product
print(f"R = {abs(R):.10f}")
