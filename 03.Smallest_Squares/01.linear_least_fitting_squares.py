import numpy as np
import matplotlib.pyplot as plt

# Get user input for x and y values
x_values = input("Enter x values separated by spaces: ")
y_values = input("Enter y values separated by spaces: ")

# Convert input strings to numpy arrays
x = np.array([float(val) for val in x_values.split()])
y = np.array([float(val) for val in y_values.split()])

# Define linear model
A = np.vstack([x, np.ones(len(x))]).T
a0, a1 = np.linalg.lstsq(A, y, rcond=None)[0]

# Displaying the fitted model
fitted_model = f"FittedModel: {a0} + {a1}x"
print(fitted_model)

# Plot the original data and the fitted line
plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, a0 + a1 * x, label=f"{a0:.2f} + {a1:.2f}x", color="orange")
plt.legend()
plt.show()

# Error calculation
absolute_error = np.sum((a0 + a1 * x - y) ** 2)
print(f"Absolute Error: {absolute_error}")
