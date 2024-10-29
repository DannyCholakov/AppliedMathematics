import numpy as np
import pandas as pd

# Даниел Чолаков ФК:2101681030  q=3 p=3

# Инициализиране на данните за таблицата
M = 1e6  # Голяма стойност M за изкуствените променливи

# Начална симплекс таблица, базирана на дадената задача
# Колони: x1, x2, x3, s1, s2, A1, RHS (дясна страна)
tableau = np.array([
    [1,  1,  2,  0,  0,  0,  4],
    [0,  1,  1, -1,  0,  1,  2],
    [2,  1,  2,  0,  1,  0,  5],
    [-3, 0, -2,  0,  0, -M, 0]
])

# Конвертиране в DataFrame за по-добра визуализация
columns = ['x1', 'x2', 'x3', 's1', 's2', 'A1', 'RHS']
index = ['x1', 'x2', 's2', 'Z']
simplex_tableau = pd.DataFrame(tableau, columns=columns, index=index)

# Показване на началната симплекс таблица
simplex_tableau

def simplex_iteration(tableau, M):
    """
    Извършва една итерация на симплекс метода върху таблицата.
    Връща обновената таблица, позицията на ключовия елемент и статуса.
    """
    # Стъпка 1: Намираме влизащата променлива (най-негативен коефициент в последния ред)
    entering_column = np.argmin(tableau[-1, :-1])
    if tableau[-1, entering_column] >= 0:
        # Оптимално решение е постигнато
        return tableau, None, None, "optimal"

    # Стъпка 2: Намираме излизащата променлива чрез минимално съотношение
    ratios = []
    for i in range(len(tableau) - 1):  # Пропускаме последния ред (целева функция)
        if tableau[i, entering_column] > 0:
            ratio = tableau[i, -1] / tableau[i, entering_column]
            ratios.append(ratio)
        else:
            ratios.append(np.inf)  # Игнориране на отрицателни или нулеви стойности

    leaving_row = np.argmin(ratios)
    if ratios[leaving_row] == np.inf:
        return tableau, None, None, "unbounded"

    # Стъпка 3: Извършване на ключовия елемент
    pivot = tableau[leaving_row, entering_column]
    tableau[leaving_row, :] /= pivot  # Нормализиране на реда с ключовия елемент

    # Обновяване на останалите редове за формиране на единичен вектор в колоната на ключовия елемент
    for i in range(len(tableau)):
        if i != leaving_row:
            row_factor = tableau[i, entering_column]
            tableau[i, :] -= row_factor * tableau[leaving_row, :]

    return tableau, entering_column, leaving_row, "continue"


# Итеративно прилагане на симплекс итерации, докато не бъде намерено оптимално решение
iteration_count = 0
status = "continue"

while status == "continue":
    tableau, entering_column, leaving_row, status = simplex_iteration(tableau, M)
    iteration_count += 1

    # Печат на междинните резултати след всяка итерация за проследяване
    print(f"Итерация {iteration_count}:")
    print(f"Индекс на влизащата променлива: {entering_column}")
    print(f"Индекс на излизащата променлива: {leaving_row}")
    print("Обновена таблица:")
    print(pd.DataFrame(tableau, columns=columns, index=index))
    print("\n")

# Краен резултат със статус оптимално или неограничено решение
status, pd.DataFrame(tableau, columns=columns, index=index)


# Итерация 1:
# Индекс на влизащата променлива: 5
# Индекс на излизащата променлива: 1
# Обновена таблица:
#      x1         x2        x3         s1   s2   A1        RHS
# x1  1.0        1.0       2.0        0.0  0.0  0.0        4.0
# x2  0.0        1.0       1.0       -1.0  0.0  1.0        2.0
# s2  2.0        1.0       2.0        0.0  1.0  0.0        5.0
# Z  -3.0  1000000.0  999998.0 -1000000.0  0.0  0.0  2000000.0
#
#
# Итерация 2:
# Индекс на влизащата променлива: None
# Индекс на излизащата променлива: None
# Обновена таблица:
#      x1         x2        x3         s1   s2   A1        RHS
# x1  1.0        1.0       2.0        0.0  0.0  0.0        4.0
# x2  0.0        1.0       1.0       -1.0  0.0  1.0        2.0
# s2  2.0        1.0       2.0        0.0  1.0  0.0        5.0
# Z  -3.0  1000000.0  999998.0 -1000000.0  0.0  0.0  2000000.0