import numpy as np
from scipy.optimize import linprog

# Дадени стойности за p и q
p = 3
q = 3

# Транспортни разходи (матрица C), стойности за p и q
C = [
    [2, q, 1, p, 6],
    [2, 6, 1, 5, 2],
    [3, 4, q, 3, 2]
]

# Ограничения за доставки (a) и търсене (b)
a = [55, 50, 10 * p]
b = [30, 40, 10 * q, 60, 20]

# Преобразуване на матрицата за минимизация
c = np.array(C).flatten()

# Създаване на ограниченията за задачата
A_eq = []
for i in range(len(a)):
    row = [0] * len(c)
    row[i * len(b):(i + 1) * len(b)] = [1] * len(b)
    A_eq.append(row)
A_eq = np.array(A_eq)
b_eq = a

A_ub = []
for j in range(len(b)):
    row = [0] * len(c)
    row[j::len(b)] = [1] * len(a)
    A_ub.append(row)
A_ub = np.array(A_ub)
b_ub = b

# Решаване на транспортната задача
print("Част (а): Решаване на транспортната задача чрез метода на линейното програмиране")
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
if result.success:
    print("Оптимално разпределение на ресурси:")
    print(result.x.reshape(len(a), len(b)))
    print(f"Минимални разходи: {result.fun}")
else:
    print("Неуспешно решение: Няма оптимално разпределение на ресурси.")
