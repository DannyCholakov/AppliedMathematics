import numpy as np

# Дадени стойности за p и q
p = 3
q = 3
tolerance = 1e-4
max_iterations = 1000

# Част (а): Дефиниране на итеративната функция за простата итерация
def iteration_function(x):
    x1 = (1 + 3*x[1] - x[2] - 2*x[3]) / 12
    x2 = (6 + 2*x[0] + 3*x[2] - x[3]) / 11
    x3 = (-3 - x[0] + 5*x[1] + 2*x[3]) / 10
    x4 = (3 - 3*x[0] - x[1] + x[2]) / 7
    return np.array([x1, x2, x3, x4])

A = np.array([
    [12, -p, 1, 2],
    [-2, q + 8, -3, 1],
    [1, -5, 10, -2],
    [3, 1, -1, -7]
])
print("Матрицата A:\n", A)

# Част (б): Приблизително решение чрез итеративен метод
print("Част (б): Приблизително решение чрез итеративен метод")
x_iter = np.zeros(4)  # Начално предположение

# Изпълнение на итеративния процес
solution_history = [x_iter.copy()]
for iteration in range(max_iterations):
    x_new_iter = iteration_function(x_iter)
    solution_history.append(x_new_iter.copy())
    # Проверка за сходимост
    if np.linalg.norm(x_new_iter - x_iter, ord=np.inf) < tolerance:
        break
    x_iter = x_new_iter

# Крайното решение, закръглено до 4 десетични знака
final_solution = np.round(x_iter, 4)
print("Крайно решение:", final_solution)

# Част (в): Проверка на точността чрез изчисляване на остатъците
print("Част (в): Проверка на точността чрез изчисляване на остатъците")
def calculate_residuals(x):
    eq1 = 12 * x[0] - p * x[1] + x[2] + 2 * x[3] - 1
    eq2 = -2 * x[0] + (q + 8) * x[1] - 3 * x[2] + x[3] - (p + q)
    eq3 = x[0] - 5 * x[1] + 10 * x[2] - 2 * x[3] + q
    eq4 = 3 * x[0] + x[1] - x[2] - 7 * x[3] - p
    return np.array([eq1, eq2, eq3, eq4])

residuals = calculate_residuals(final_solution)
print("Остатъци:")
for i, res in enumerate(residuals, 1):
    print(f"  Остатък {i}: {res:.6f}")

# Част (г): Проверка на условието за сходимост чрез диагонална доминантност
print("Част (г): Проверка на условието за сходимост чрез диагонална доминантност")


# Проверка дали матрицата е диагонално доминантна
is_diagonally_dominant = all(
    abs(A[i, i]) > sum(abs(A[i, j]) for j in range(len(A)) if j != i)
    for i in range(len(A))
)
print("Матрицата е диагонално доминантна:", is_diagonally_dominant)

# Допълнително: Изчисляване на спектралния радиус за проверка на сходимостта
eigenvalues = np.linalg.eigvals(A)
spectral_radius = max(abs(eigenvalues))
print("Спектрален радиус:", spectral_radius)
