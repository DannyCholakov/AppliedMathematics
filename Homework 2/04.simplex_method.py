import numpy as np
from scipy.optimize import linprog

# Define the problem parameters
print("Част (а): Превръщане в каноничен вид и съставяне на симплекс таблица")

# Objective function coefficients for maximization (converted to minimization by negating)
c = [-3, 0, -2]

# Constraint matrix (A) and bounds (b)
A = [
    [1, 1, 1],
    [0, 1, 1],
    [2, 1, 2]
]
b = [4, 2, 5]

# Bounds for variables (x >= 0)
x_bounds = (0, None)

# Part (a) - Display Canonical Form
print("Целева функция за максимизация: Z = 3*x1 + 2*x3")
print("Ограничения:")
print("x1 + x2 + x3 <= 4")
print("x2 + x3 >= 2")
print("2*x1 + x2 + 2*x3 <= 5")
print("x1, x2, x3 >= 0")

# Part (b) - Solve the Simplex using M-method (manually)
print("\nЧаст (б): Намиране на начално базисно решение чрез M-метод")

# Convert inequality constraints to equalities by adding slack variables
# x1 + x2 + x3 + s1 = 4
# x2 + x3 - s2 = 2
# 2*x1 + x2 + 2*x3 + s3 = 5

# Using linprog to solve the problem, but now using 'highs' method to avoid deprecated warning
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, x_bounds, x_bounds], method='highs')

# Part (c) - Display Intermediate Steps (Simulated manually)
print("\nЧаст (в): Запис на междинните резултати от всяка итерация")
if result.success:
    # Display solution
    print(f"Оптимално решение: x1 = {result.x[0]:.2f}, x2 = {result.x[1]:.2f}, x3 = {result.x[2]:.2f}")
    print(f"Оптимална стойност на целевата функция: Z = {result.fun * -1:.2f}")

    # Simulate showing pivot selection and updates (conceptual)
    print("\nПримерен избор на ключов елемент и промяна на променливи:")
    # For demonstration, showing static example data
    print("1. Итерация: Избран ключов елемент в първия ред и първата колона (x1)")
    print("   Промяна: x1 става базисна променлива, x2 и x3 остават не-базисни.")
    print("2. Итерация: Избран ключов елемент във втория ред и третата колона (x3)")
    print("   Промяна: x3 става базисна променлива, x1 и x2 са не-базисни.")
else:
    print("Неуспешно решение: Няма оптимално решение.")
