import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# Задача 1: Функция за решаване на система от линейни уравнения чрез итеративен метод
def iterative_method(a, b, eps=0.0001, max_iter=100):
    x = np.zeros(len(b), dtype=float)
    d = b / np.diag(a)
    c = np.eye(len(a)) - (a / np.diag(a).reshape(-1, 1))

    print("c =")
    print(np.array2string(c, formatter={'float_kind': lambda x: f"{x: .6f}"}))
    print("d =")
    print(np.array2string(d, formatter={'float_kind': lambda x: f"{x: .6f}"}))

    err = float('inf')
    k = 0
    while err >= eps and k < max_iter:
        x_new = c @ x + d
        err = np.max(np.abs(x_new - x))
        x = x_new
        print(f"k= {k+1}, x = [{', '.join([f'{xi: .6f}' for xi in x])}], текуща грешка err = {err:.6f}")
        k += 1

    norm_c_inf = np.max(np.sum(np.abs(c), axis=1))
    norm_c_one = np.max(np.sum(np.abs(c), axis=0))
    norm_c_fro = np.sqrt(np.sum(c**2))
    print(f"Нормите на преобразуваната матрица са : ({norm_c_inf:.6f}, {norm_c_one:.6f}, {norm_c_fro:.6f})")

# Задача 2: Функции за Нютонов метод
def f(x):
    return np.sin(7 * x + np.sqrt(7 + x**2)) - 7 * x + np.log(2 + x) + 4

def df(x):
    return (7 * np.cos(7 * x + np.sqrt(7 + x**2)) +
            (x / np.sqrt(7 + x**2)) * np.cos(7 * x + np.sqrt(7 + x**2)) -
            7 + 1 / (2 + x))

def ddf(x):
    h = 1e-5
    return (df(x + h) - df(x - h)) / (2 * h)

# Задача 3: Метод на Нютон
def newton_method(x0, eps=0.0001, max_iter=100):
    print("Начално приложение")
    for n in range(max_iter):
        f_x0 = f(x0)
        df_x0 = df(x0)

        if np.isnan(f_x0) or np.isnan(df_x0):
            print("Намерена е невалидна стойност по време на изчислението. Изход.")
            break

        if df_x0 == 0:
            print("Производната е нула. Няма решение.")
            break

        x1 = x0 - f_x0 / df_x0
        err = abs(x1 - x0)
        print(f"m = {n}, x = {x0:.10f}, f(x0) = {f_x0:.10f}, грешка = {err:.10f}")

        if err < eps:
            print(f"Отговор х = {x1:.10f} с грешка = {err:.10f}")
            break

        x0 = x1
    else:
        print("Достигнат е максималният брой итерации. Решението може да не е сближено.")

# Задача 4: Метод за полиномиално апроксимиране по метода на най-малките квадрати
def least_squares_method(x, y, z):
    A1 = np.vstack([np.ones(len(x)), x]).T
    coef1, _, _, _ = lstsq(A1, y, rcond=None)
    P1 = lambda t: coef1[0] + coef1[1] * t
    error1 = np.sqrt(np.sum((P1(x) - y) ** 2))
    P1_z = P1(z)

    A2 = np.vstack([np.ones(len(x)), x, x**2]).T
    coef2, _, _, _ = lstsq(A2, y, rcond=None)
    P2 = lambda t: coef2[0] + coef2[1] * t + coef2[2] * t**2
    error2 = np.sqrt(np.sum((P2(x) - y) ** 2))
    P2_z = P2(z)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Точки на данните")
    x_plot = np.linspace(0, 1, 100)
    plt.plot(x_plot, P1(x_plot), label=f"Модел от първа степен: y = {coef1[0]:.2f} + {coef1[1]:.2f} * x", color="red")
    plt.plot(x_plot, P2(x_plot), label=f"Модел от втора степен: y = {coef2[0]:.2f} + {coef2[1]:.2f} * x + {coef2[2]:.2f} * x^2", color="green")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Приближение по метода на най-малките квадрати")
    plt.grid(True)
    plt.show()

    print("Модел от първа степен:")
    print(f"Уравнение: P1(t) = {coef1[0]:.2f} + {coef1[1]:.2f} * t")
    print(f"Абсолютна грешка: {error1:.2f}")
    print(f"Приближение при z = {z}: P1(z) = {P1_z:.2f}")

    print("\nМодел от втора степен:")
    print(f"Уравнение: P2(t) = {coef2[0]:.2f} + {coef2[1]:.2f} * t + {coef2[2]:.2f} * t^2")
    print(f"Абсолютна грешка: {error2:.2f}")
    print(f"Приближение при z = {z}: P2(z) = {P2_z:.2f}")

# Тестови данни за всяка от задачите
a_test = np.array([
    [12., -3., 1., 2.],
    [-2., 11., -3., 1.],
    [1., -5., 13., -2.],
    [3., 1., -1., -7.]
])
b_test = np.array([1, 6, -3, 3])

x_test = np.array([0, 0.3, 0.63, 0.23, 0.35, 0.46, 0.71, 0.97])
y_test = np.array([2.01, 2.47, 2.80, 2.87, 3.17, 3.44, 4.56, 3.33])
z_test = 0.101 * 3

# Извикване на функции
def show_task_choices():
    print("Моля, изберете задача, която искате да решите:")
    print("1.Нютонов метод за намиране на корени")
    print("2.Метод за решаване на система от линейни уравнения чрез итеративен метод")
    print("3.Метод на Нютон")
    print("4.Полиномиално апроксимиране по метода на най-малките квадрати")

show_task_choices()

choice = input("Въведете номера на задачата, която искате да решите: ")

if choice == '1':
    newton_method(1)
elif choice == '2':
    iterative_method(a_test, b_test)
elif choice == '3':
    newton_method(1)
elif choice == '4':
    least_squares_method(x_test, y_test, z_test)
else:
    print("Невалиден избор! Моля, изберете валиден номер на задача.")
