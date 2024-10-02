import math
from sympy import symbols, diff, sin, cos, exp
import matplotlib.pyplot as plt
import numpy as np

# Define x as a symbol for sympy
x = symbols('x')
function_string = input("Enter your function: ")
function_transform = eval(function_string)


# Define the function using sympy's exp and polynomial
def f(x):
    return function_transform

    # Example function:
    # x + exp(x)
    # x-sin(x)-0.25
    # exp(x)-x**2 - 2*x - 2


# Derivative of the function f(x)
f_prime = diff(f(x), x)
f_double_prime = diff(f_prime, x)

# Convert to Python functions for numeric evaluation
f_numeric = lambda x_val: float(f(x).subs(x, x_val))
f_prime_numeric = lambda x_val: float(f_prime.subs(x, x_val))
f_double_prime_numeric = lambda x_val: float(f_double_prime.subs(x, x_val))


# Function to plot the initial graph (Preview) for f(x)
def plot_preview():
    # Use a default range from -5 to 5 or adjust as needed
    x_vals_preview = np.linspace(-5, 5, 100)
    f_vals_preview = [f_numeric(val) for val in x_vals_preview]

    plt.figure(figsize=(8, 8))  # Set figure size
    plt.plot(x_vals_preview, f_vals_preview, label="f(x)", color="blue", linewidth=2)

    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis
    plt.title('Preview of f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Preview the graph for the user before they select an interval
plot_preview()

# Interval values
a = float(input('Enter interval A: '))
b = float(input('Enter interval B: '))
tolerance = int(input('Enter interval Tolerance: '))

# Define a larger range for the whole plot
x_vals_full = np.linspace(-10, 10, 1000)

# Compute y values for f(x), f'(x), and f''(x) over the full range
f_vals_full = [f_numeric(val) for val in x_vals_full]
f_prime_vals_full = [f_prime_numeric(val) for val in x_vals_full]
f_double_prime_vals_full = [f_double_prime_numeric(val) for val in x_vals_full]

# Define the range for the colored lines (between a and b)
x_vals_range = np.linspace(a, b, 100)
f_vals_range = [f_numeric(val) for val in x_vals_range]
f_prime_vals_range = [f_prime_numeric(val) for val in x_vals_range]
f_double_prime_vals_range = [f_double_prime_numeric(val) for val in x_vals_range]


def plot():
    # Plot the full graph
    plt.figure(figsize=(10, 10))  # Make the plot square for equal axes

    # Plot f(x) over the full range and then highlight it between a and b
    plt.plot(x_vals_full, f_vals_full, label="f(x)", color="gray", linestyle='--')
    plt.plot(x_vals_range, f_vals_range, label="f(x) in range", color="blue", linewidth=2)

    # Plot f/'(x) over the full range and then highlight it between a and b
    plt.plot(x_vals_full, f_prime_vals_full, label="f'(x)", color="gray", linestyle='--')
    plt.plot(x_vals_range, f_prime_vals_range, label="f'(x) in range", color="red", linewidth=2)

    # Plot f''(x) over the full range and then highlight it between a and b
    plt.plot(x_vals_full, f_double_prime_vals_full, label="f''(x)", color="gray", linestyle='--')
    plt.plot(x_vals_range, f_double_prime_vals_range, label="f''(x) in range", color="green", linewidth=2)

    # Set plot limits and labels
    plt.xlim(-20, 20)  # Adjust the x-range as needed
    plt.ylim(-20, 20)  # Adjust the y-range as needed
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio for x and y axes
    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis
    plt.title(f'Plot of f(x) = {f(x)}')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    # Add legend and show the plot
    plt.legend()
    plt.grid(True)
    plt.show()


# Check if the function has different signs at the ends of the interval
if f_numeric(a) * f_numeric(b) > 0:
    print("Bad interval!")  # Bad interval
else:
    # Initial guess for x0
    x0 = a if f_numeric(a) * f_prime_numeric(a) > 0 else b
    print('Initial guess x0 =', x0)

    # Set precision and initial error
    eps = 10 ** - tolerance  # Tolerance for stopping criterion
    n = 0  # Iteration counter
    err = math.inf  # Initialize error

    # Newton-Raphson loop
    while True:
        x1 = x0 - f_numeric(x0) / f_prime_numeric(x0)  # Newton's update rule
        print(f'n = {n}, x = {x0:9f}, f[x] = {f_numeric(x0):9e}, err = {err:9e}')
        err = abs(x1 - x0)  # Update error
        # Update x0 and increment the counter
        n += 1
        x0 = x1  # Update x0
        if err <= eps:  # Exit condition after printing
            break

    # Final iteration
    print(f'n = {n}, x = {x0:9f}, f[x] = {f_numeric(x0):9e}, err = {err:9e}')
    print(f'Result x* ~ {x0:6f} with error {err:9e}')
    plot()
