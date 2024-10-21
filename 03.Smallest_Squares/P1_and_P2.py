import numpy as np
import matplotlib.pyplot as plt


def get_data_from_user():
    """
    This function gets the data points (x, y) from the user.
    """
    x = input("Enter the x values (separated by spaces): ")
    y = input("Enter the y values (separated by spaces): ")

    # Convert the string input to lists of floats
    x = np.array([float(i) for i in x.split()])
    y = np.array([float(i) for i in y.split()])

    return x, y


def linear_fit(x, y):
    """
    Performs a least squares fit for a polynomial of degree 1.
    """
    # Create the design matrix for linear fit (degree 1)
    A = np.vstack([x, np.ones(len(x))]).T

    # Solve for the coefficients using least squares
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    # Create the fitted line
    P1 = coeffs[0] * x + coeffs[1]

    # Plot the data points and the fitted line
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x, P1, 'r', label='Fitted line P1*')
    plt.legend()
    plt.show()

    # Print the coefficients
    print(f"Linear Fit Coefficients: a1 = {coeffs[0]}, a0 = {coeffs[1]}")


def quadratic_fit(x, y):
    """
    Performs a least squares fit for a polynomial of degree 2.
    """
    # Create the design matrix for quadratic fit (degree 2)
    A = np.vstack([x ** 2, x, np.ones(len(x))]).T

    # Solve for the coefficients using least squares
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    # Create the fitted polynomial
    P2 = coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2]

    # Plot the data points and the fitted polynomial
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x, P2, 'r', label='Fitted polynomial P2*')
    plt.legend()
    plt.show()

    # Print the coefficients
    print(f"Quadratic Fit Coefficients: a2 = {coeffs[0]}, a1 = {coeffs[1]}, a0 = {coeffs[2]}")


def main():
    """
    Main function that interacts with the user and applies the appropriate least squares method.
    """
    print("Welcome to the Least Squares Fitting Program!")

    # Get user data
    x, y = get_data_from_user()

    # Ask the user for the degree of polynomial to fit
    degree = int(input("Enter the degree of the polynomial (1 for linear, 2 for quadratic): "))

    # Perform the appropriate fit
    if degree == 1:
        linear_fit(x, y)
    elif degree == 2:
        quadratic_fit(x, y)
    else:
        print("Invalid degree choice. Please enter 1 for linear or 2 for quadratic.")


# Run the main function
if __name__ == "__main__":
    main()
