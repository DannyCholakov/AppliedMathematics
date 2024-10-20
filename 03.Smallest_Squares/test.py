import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Input data points
x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y = np.array([-2.0, -1.60901699, -0.909017, -0.090983, 0.609017, 1.0,
              1.00901699, 0.709017, 0.290983, -0.00901699, 0.0])

# Fit polynomials of degree 1 and 2 using least squares
P1 = np.polyfit(x, y, 1)  # First-degree polynomial
P2 = np.polyfit(x, y, 2)  # Second-degree polynomial

# Define polynomial functions
def poly1(x):
    return P1[0] * x + P1[1]

def poly2(x):
    return P2[0] * x**2 + P2[1] * x + P2[2]

# Prepare a fine grid of points for plotting the polynomials
t_values = np.linspace(-1, 1, 100)

# Plot original data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the first-degree polynomial
plt.plot(t_values, poly1(t_values), label='First Degree Polynomial')

# Plot the second-degree polynomial
plt.plot(t_values, poly2(t_values), label='Second Degree Polynomial')

# Calculate Absolute Error (RMSE) for both polynomials
def absolute_error(poly_func, x, y):
    N = len(x)
    error = np.sqrt(np.sum((poly_func(x) - y) ** 2) / N)
    return error

# Calculate Absolute Error (RMSE) for both polynomials
AbsError1 = absolute_error(poly1, x, y)
AbsError2 = absolute_error(poly2, x, y)

# Print results
print("Стойностите на x са:", x)
print("Стойностите на y са:", y)
print(f"Полиномът от първа степен е: {P1[1]:.8f} + {P1[0]:.6f}·x")
print("Абсолютна грешка (първа степен):", AbsError1)
print(f"Полиномът от втора степен е: {P2[2]:.8f} + {P2[1]:.6f}·x + {P2[0]:.6f}·x²")
print("Абсолютна грешка (втора степен):", AbsError2)

# Show plot
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fit and Errors')
plt.grid()
plt.show()