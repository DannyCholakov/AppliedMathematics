import numpy as np
from scipy.optimize import linprog

# Даниел Чолаков ФК:2101681030  q=3 p=3

# Оригинална матрица на разходите с добавен фиктивен ред (последен ред)
C = np.array([
    [2, 3, 1, 3, 6],
    [2, 6, 1, 5, 2],
    [3, 4, 3, 3, 2],
    [0, 0, 0, 0, 0]  # Фиктивен ред с нулеви разходи
])

# Вектор на предлагането с добавено фиктивно предлагане (последен елемент)
supply = np.array([55, 50, 30, 45])

# Вектор на търсенето остава същият
demand = np.array([30, 40, 30, 60, 20])

# Целева функция (разпъната матрица на разходите)
c = C.flatten()

# Ограничения за предлагането (всеки ред в матрицата на разходите)
A_eq_supply = np.kron(np.eye(supply.size), np.ones(demand.size))
b_eq_supply = supply

# Ограничения за търсенето (всяка колона в матрицата на разходите)
A_eq_demand = np.kron(np.ones(supply.size), np.eye(demand.size))
b_eq_demand = demand

# Комбиниране на ограниченията за равенство
A_eq = np.vstack([A_eq_supply, A_eq_demand])
b_eq = np.hstack([b_eq_supply, b_eq_demand])

# Граници за променливите (неотрицателни стойности)
bounds = [(0, None)] * (supply.size * demand.size)

# Решаване на транспортната задача с линеарно програмиране
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# Извличане на решението като матрица
solution = result.x.reshape(C.shape)

# Показване на оптималното разпределение и минималната обща цена
print("Оптимален транспортен план:")
print(solution)
print("\nОбщ минимум:", result.fun)

# Оптимален транспортен план:
# [[10.  0.  0. 45.  0.]
#  [20.  0. 30.  0.  0.]
#  [ 0.  0.  0. 10. 20.]
#  [ 0. 40.  0.  5.  0.]]
#
# Общ минимум: 295.0