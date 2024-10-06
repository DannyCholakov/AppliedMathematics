import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Function to evaluate the input expression safely
def safe_eval(expr, t):
    return eval(expr)

# Input function f(t)
function_input = input("Enter the function f(t) (use 't' as variable, e.g. 'math.sqrt(t)', 't**(1/3)', '1/(t**2 + 3)': ")

# Define function for approximation
def f(t):
    return safe_eval(function_input, t)

# Get the input for xx values and point (u)
xx = list(map(float, input("Enter the xx values separated by spaces: ").split()))
u = float(input("Enter the point (u) where you want to approximate the value: "))
n = len(xx)
yy = [f(xx[i]) for i in range(n)]

# Define the Lagrange interpolation polynomial
def L(t):
    result = 0
    for i in range(n):
        term = yy[i]
        for j in range(n):
            if i != j:
                term *= (t - xx[j]) / (xx[i] - xx[j])
        result += term
    return result

# Plot
t_values = np.linspace(min(xx) - 1, max(xx) + 1, 400)  # Range based on input xx values
L_values = [L(t) for t in t_values]
plt.scatter(xx, yy, color='blue', label='Data Points')
plt.plot(t_values, L_values, color='red', label='L(t)', linewidth=2)
plt.title('Lagrange Interpolation')
plt.xlabel('xx values')
plt.ylabel('yy values')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()

# Print xx and yy
print("xx values:", xx)
print("yy values:", yy)

# Symbolic calculations
t_sym = sp.symbols('t')
function_input = function_input.replace("math.", "sp.")

try:
    f_sym = eval(function_input.replace("t", "t_sym"))
except Exception as e:
    print("Error in evaluating function:", e)
    exit(1)

# Calculate the Lagrange polynomial L(t)
L_sym = sum(yy[i] * sp.prod((t_sym - xx[j]) / (xx[i] - xx[j]) for j in range(n) if i != j) for i in range(n))

# Print L(t)
print(f"L(t) = {sp.expand(L_sym)}")

# Calculate L(u) and the exact value
approximation_at_u = L(u)
exact_value_at_u = f(u)
print(f"L({u}) = {approximation_at_u:.6f}, точна стойност = {exact_value_at_u:.6f}")

# Calculate the n-th derivative of f(t)
n_derivative = sp.diff(f_sym, t_sym, n)

# Find critical points of the derivative (not necessary for this specific case)
critical_points = sp.solve(n_derivative, t_sym)
# We can ignore the critical points since we're not considering endpoints

# Evaluate the n-th derivative at the points in xx
M = max([abs(n_derivative.evalf(subs={t_sym: xx[i]})) for i in range(n)], default=0)

# Print M
print(f"Mn+1 = {M:.5f}")

# Calculate the remainder R
R = (M / math.factorial(n)) * np.prod([abs(u - xx[i]) for i in range(n)])
print(f"R = {abs(R):.10f}")
