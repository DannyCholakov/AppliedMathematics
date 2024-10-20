import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Get user input for x and y values
x_values = input("Enter x values separated by spaces: ")
y_values = input("Enter y values separated by spaces: ")

# Convert input strings to numpy arrays
x = np.array([float(val) for val in x_values.split()])
y = np.array([float(val) for val in y_values.split()])

# User input for the function
user_function = input("Enter the function to fit (use 'a0', 'a1', 'a2', etc. as parameters, and 'x' as the variable):\n")

# Convert the string into a Python function
def func(x, *params):
    param_dict = {f'a{i}': params[i] for i in range(len(params))}
    return eval(user_function, {"np": np, "x": x}, param_dict)

# Guess the number of parameters (count occurrences of 'a')
num_params = user_function.count('a')

# Curve fitting
popt, pcov = curve_fit(func, x, y, p0=[1] * num_params)

# Displaying the fitted model
param_str = " + ".join([f"{popt[i]}*x^{i}" for i in range(len(popt))])
fitted_model = f"FittedModel: {param_str}"
print(fitted_model)

# Plot the original data and the fitted model
plt.scatter(x, y, color="blue", label="Data Points")
t_vals = np.linspace(min(x), max(x), 100)
plt.plot(t_vals, func(t_vals, *popt), label=fitted_model, color="orange")
plt.legend()
plt.show()

# Error calculation
absolute_error = np.sum((func(x, *popt) - y) ** 2)
print(f"Absolute Error: {absolute_error}")
