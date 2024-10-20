import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import curve_fit

# Function for user input
def user_input():
    print("Choose a function for f(x):")
    print("1. Cos(π * x) + x")
    print("2. Custom function")
    choice = int(input("Enter choice (1 or 2): "))

    if choice == 1:
        f = lambda x: np.cos(np.pi * x) + x
    else:
        formula = input("Enter a custom function in terms of 'x' (use numpy functions, e.g., np.sqrt(x)): ")
        f = eval(f"lambda x: {formula}")

    h = float(input("Enter step size (h): "))
    x_start = float(input("Enter start value for x: "))
    x_end = float(input("Enter end value for x: "))

    return f, h, x_start, x_end


# Calculate x and y values based on the function
def generate_data(f, h, x_start, x_end):
    x_vals = np.arange(x_start, x_end + h, h)
    y_vals = np.array([f(x) for x in x_vals])

    print("x values:", x_vals)
    print("y values:", y_vals)
    return x_vals, y_vals


# First-degree polynomial fitting
def first_degree_polynomial(x, y):
    n = len(x)
    A1 = np.array([[n, np.sum(x)], [np.sum(x), np.sum(x ** 2)]])
    b1 = np.array([np.sum(y), np.sum(x * y)])

    # Solve the linear system
    s1 = solve(A1, b1)
    P1 = lambda t: s1[0] + s1[1] * t

    # Calculate the absolute error
    absolute_error_1 = np.sqrt(np.sum((P1(x) - y) ** 2))

    return P1, absolute_error_1


# Second-degree polynomial fitting
def second_degree_polynomial(x, y):
    n = len(x)
    A2 = np.array([[n, np.sum(x), np.sum(x ** 2)],
                   [np.sum(x), np.sum(x ** 2), np.sum(x ** 3)],
                   [np.sum(x ** 2), np.sum(x ** 3), np.sum(x ** 4)]])
    b2 = np.array([np.sum(y), np.sum(x * y), np.sum((x ** 2) * y)])

    # Solve the linear system
    s2 = solve(A2, b2)
    P2 = lambda t: s2[0] + s2[1] * t + s2[2] * t ** 2

    # Calculate the absolute error
    absolute_error_2 = np.sqrt(np.sum((P2(x) - y) ** 2))

    return P2, absolute_error_2


# Fit custom function using curve fitting
def custom_curve_fit(x, y):
    user_function = input("Enter the function to fit (use 'a0', 'a1', 'a2', etc. as parameters, and 'x' as the variable):\n")

    # Convert the string into a Python function
    def func(x, *params):
        param_dict = {f'a{i}': params[i] for i in range(len(params))}
        return eval(user_function, {"np": np, "x": x}, param_dict)

    # Guess the number of parameters (count occurrences of 'a')
    num_params = user_function.count('a')
    if num_params == 0:
        raise ValueError("No parameters 'a0', 'a1', etc. found in the function.")

    # Curve fitting
    popt, _ = curve_fit(func, x, y, p0=[1] * num_params)

    # Displaying the fitted model
    param_str = " + ".join([f"{popt[i]}*x^{i}" for i in range(len(popt))])
    fitted_model = f"Fitted Model: {param_str}"
    print(fitted_model)

    return func, popt


# Plot the results
def plot_results(x, y, P1, P2, P_custom, x_start, x_end, popt):
    plt.scatter(x, y, color='red', label='Data Points')

    t_vals = np.linspace(x_start, x_end, 500)

    # Plot first-degree polynomial
    plt.plot(t_vals, P1(t_vals), label='1st Degree Polynomial', color='blue')

    # Plot second-degree polynomial
    plt.plot(t_vals, P2(t_vals), label='2nd Degree Polynomial', color='green')

    # Plot custom function if fitted
    plt.plot(t_vals, P_custom(t_vals, *popt), label='Custom Fit', color='orange')

    plt.legend()
    plt.show()


def main():
    # User input for the function and step size
    f, h, x_start, x_end = user_input()

    # Generate x and y values
    x, y = generate_data(f, h, x_start, x_end)

    # First-degree polynomial
    P1, absolute_error_1 = first_degree_polynomial(x, y)
    print("First-degree polynomial: P1(t) = {:.4f} + {:.4f} * t".format(P1(0), P1(1) - P1(0)))
    print("Absolute Error for first-degree polynomial: {:.4f}".format(absolute_error_1))

    # Second-degree polynomial
    P2, absolute_error_2 = second_degree_polynomial(x, y)
    print("Second-degree polynomial: P2(t) = {:.4f} + {:.4f} * t + {:.4f} * t^2".format(P2(0), P2(1) - P2(0),
                                                                                        (P2(2) - P2(1)) / (1 - 0) ** 2))
    print("Absolute Error for second-degree polynomial: {:.4f}".format(absolute_error_2))

    # Custom function fitting
    P_custom, popt = custom_curve_fit(x, y)

    # Plot results
    plot_results(x, y, P1, P2, P_custom, x_start, x_end, popt)


if __name__ == "__main__":
    main()
