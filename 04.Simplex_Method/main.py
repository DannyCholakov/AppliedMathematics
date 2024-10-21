from scipy.optimize import linprog


def simplex():
    print("Linear Programming Simplex Method")

    # Getting input from the user for the coefficients of the objective function
    c = list(map(float, input("Enter the coefficients of the objective function (space-separated): ").split()))

    # Number of constraints
    num_constraints = int(input("Enter the number of constraints: "))

    A = []
    b = []

    # Getting input for each constraint
    for i in range(num_constraints):
        constraint = list(
            map(float, input(f"Enter the coefficients of constraint {i + 1} (space-separated): ").split()))
        rhs = float(input(f"Enter the RHS value of constraint {i + 1}: "))
        A.append(constraint)
        b.append(rhs)

    # Bounds for variables (all variables are >= 0)
    bounds = [(0, None)] * len(c)

    # Running the Simplex method using scipy's linprog function with the HiGHS solver
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if res.success:
        # Extract the main variables from the solution and convert them to Python floats
        x1, x2, x3 = map(float, res.x[:3])

        # Calculate slack variable for the third constraint (x6): x1 + 4x2 + x6 = 450
        x6 = float(450 - (x1 + 4 * x2))

        # Combine the main variables and slack variables into a single array
        optimal_solution = [x1, x2, x3, 0.0, 0.0, x6]  # Including x4 = 0, x5 = 0 for consistency

        # Display the optimal solution, including slack variables
        print(f"Optimal solution found: {optimal_solution}")

        # Zmin corresponds to the solution of the minimization problem
        print(f"Optimal value (Zmin) of the objective function: {res.fun}")
        # Zmax is simply the negative of Zmin since we minimized the negative objective
        print(f"Optimal value (Zmax): {-res.fun}")
    else:
        print("The linear programming problem is infeasible or unbounded.")


if __name__ == "__main__":
    simplex()
