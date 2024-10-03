import numpy as np

# Function to convert input to a list of floats
def input_to_float_list(prompt):
    return [float(x) for x in input(prompt).split(',')]

# Take user input for the size of the matrix (n x n)
n = int(input("Enter the size of the matrix (n x n): "))

# Take user input for the matrix A
A = []
print(f"Enter the matrix A ({n}x{n}) row by row, with values separated by commas.")
for i in range(n):
    row = input_to_float_list(f'Row {i+1}: ')
    A.append(row)

# Convert A to a numpy array
A = np.array(A)

# Take user input for the vector b
print(f"Enter the vector b ({n} elements) separated by commas.")
b = np.array(input_to_float_list('Vector b: '))

# Initial guess for x
x = np.zeros(len(b))

# Parameters
tolerance = int(input("Tolerance: "))
eps = 10**-tolerance  # tolerance for the stopping criterion
err = float('inf')  # initialize the error as infinity
max_iter = 1000  # maximum number of iterations to avoid infinite loop

# Compute D and C matrices for the iteration
D = b / np.diag(A)
C = np.identity(n) - (A / np.diag(A)[:, np.newaxis])

# Print the C and D matrices
print("C matrix:\n", C)
print("D vector:\n", D)

# Function to calculate the norms
def calculate_norms(C):
    # Row Norm (Maximum Row Sum)
    row_norm = np.max(np.sum(np.abs(C), axis=1))

    # Column Norm (Maximum Column Sum)
    col_norm = np.max(np.sum(np.abs(C), axis=0))

    # Frobenius Norm (2-norm of the matrix)
    frobenius_norm = np.sqrt(np.sum(C ** 2))

    return row_norm, col_norm, frobenius_norm

# Iteration counter
k = 0

while err >= eps and k < max_iter:
    # Perform the iteration
    x_new = np.dot(C, x) + D

    # Calculate the error
    err = np.max(np.abs(x_new - x))

    # Update x and iteration counter
    x = x_new
    k += 1

    print(f"Iteration {k}: x = {x}, Error = {err}")

# Print final result
print(f"\nFinal approximation after {k} iterations:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")

# Calculate and print the norms of matrix C
row_norm, col_norm, frobenius_norm = calculate_norms(C)
print(f"\nRow Norm (Max Row Sum): {row_norm:.6f}")
print(f"Column Norm (Max Column Sum): {col_norm:.6f}")
print(f"Frobenius Norm: {frobenius_norm:.6f}")

# Compare with the built-in solution
x_exact = np.linalg.solve(A, b)
print("\nExact solution using np.linalg.solve:")
for i in range(n):
    print(f"x{i+1} = {x_exact[i]:.3f}")

