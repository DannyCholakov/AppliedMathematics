import numpy as np

# Функция за конвертиране на вход в списък от числа с плаваща запетая
def input_to_float_list(prompt):
    return [float(x) for x in input(prompt).split(',')]

# Вземане на вход за размера на матрицата (n x n)
n = int(input("Въведете размера на матрицата (n x n): "))

# Вземане на вход за матрицата A
A = []
print(f"Въведете матрицата A ({n}x{n}) ред по ред, като стойностите са разделени със запетая.")
for i in range(n):
    row = input_to_float_list(f'Ред {i+1}: ')
    A.append(row)

# Конвертиране на A в numpy масив
A = np.array(A)

# Вземане на вход за вектора b
print(f"Въведете вектора b ({n} елемента), разделени със запетая.")
b = np.array(input_to_float_list('Вектор b: '))

# Начална предположена стойност за x
x = np.zeros(len(b))

# Параметри
tolerance = int(input("Толерантност: "))
eps = 10**-tolerance  # толеранс за критерия за спиране
err = float('inf')  # инициализиране на грешката като безкрайност
max_iter = 1000  # максимален брой итерации, за да се избегне безкраен цикъл

# Изчисляване на матриците D и C за итерацията
D = b / np.diag(A)
C = np.identity(n) - (A / np.diag(A)[:, np.newaxis])

# Печат на матриците C и D
print("Матрица C:\n", C)
print("Вектор D:\n", D)

# Функция за изчисляване на нормите
def calculate_norms(C):
    # Норма на реда (максимална сума по редове)
    row_norm = np.max(np.sum(np.abs(C), axis=1))

    # Норма на колоните (максимална сума по колони)
    col_norm = np.max(np.sum(np.abs(C), axis=0))

    # Фробениусова норма (2-норма на матрицата)
    frobenius_norm = np.sqrt(np.sum(C ** 2))

    return row_norm, col_norm, frobenius_norm

# Брояч на итерации
k = 0

while err >= eps and k < max_iter:
    # Изпълнение на итерацията
    x_new = np.dot(C, x) + D

    # Изчисляване на грешката
    err = np.max(np.abs(x_new - x))

    # Обновяване на x и брояча на итерациите
    x = x_new
    k += 1

    print(f"Итерация {k}: x = {x}, Грешка = {err}")

# Печат на крайния резултат
print(f"\nКрайната апроксимация след {k} итерации:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")

# Изчисляване и печат на нормите на матрица C
row_norm, col_norm, frobenius_norm = calculate_norms(C)
print(f"\nНорма на реда (Максимална сума на ред): {row_norm:.6f}")
print(f"Норма на колоните (Максимална сума на колона): {col_norm:.6f}")
print(f"Фробениусова норма: {frobenius_norm:.6f}")

# Сравняване с вграденото решение
x_exact = np.linalg.solve(A, b)
print("\nТочно решение с np.linalg.solve:")
for i in range(n):
    print(f"x{i+1} = {x_exact[i]:.3f}")
