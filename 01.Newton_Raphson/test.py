import numpy as np

# Define the number of variables
n = 4  # As we have 4 equations

# Define the iteration matrices D and C based on rearranged equations
D = np.array([
    0, 6, -1, 0.5  # The constants on the right-hand side of the equations
])

# Create C matrix based on rearranged equations
C = np.array([
    [0, 1/5, -2/5, -1/5],
    [1/10, 0, 2/10, -3/10],
    [1/10, -4/10, 0, 0],
    [3/8, 0, 1/8, 0]
])

# Initial guess
x = np.zeros(n)  # Start with the zero vector

# Parameters
tolerance = 10**-4  # Set tolerance as specified
max_iter = 10  # Maximum number of iterations as per exercise 4

# Iteration counter
k = 0
err = float('inf')  # Initialize error

# Iteration loop
while err >= tolerance and k < max_iter:
    x_new = np.dot(C, x) + D
    err = np.max(np.abs(x_new - x))  # Calculate the maximum error
    x = x_new  # Update the solution
    k += 1
    print(f"Iteration {k}: x = {x}, Error = {err}")

# Final result
print(f"\nFinal approximation after {k} iterations:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.4f}")

# Print final error
print(f"Final error after {k} iterations: {err:.6f}")
