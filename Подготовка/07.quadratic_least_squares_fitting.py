import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt


# Функция за въвеждане на данни от потребителя
def user_input():
    print("Изберете функция за f(x):")
    print("1. Cos(π * x) + x")
    print("2. Потребителска функция")
    choice = int(input("Въведете избор (1 или 2): "))

    if choice == 1:
        f = lambda x: np.cos(np.pi * x) + x
    else:
        formula = input("Въведете потребителска функция чрез 'x' (използвайте функции от numpy, например np.sqrt(x)): ")
        f = eval(f"lambda x: {formula}")

    h = float(input("Въведете стъпка (h): "))
    x_start = float(input("Въведете началната стойност за x: "))
    x_end = float(input("Въведете крайната стойност за x: "))

    return f, h, x_start, x_end


# Изчисляване на x и y стойности въз основа на функцията
def generate_data(f, h, x_start, x_end):
    x_vals = np.arange(x_start, x_end + h, h)
    y_vals = np.array([f(x) for x in x_vals])

    print("x стойности:", x_vals)
    print("y стойности:", y_vals)
    return x_vals, y_vals


# Полиномоализация от първа степен
def first_degree_polynomial(x, y):
    n = len(x)
    A1 = np.array([[n, np.sum(x)], [np.sum(x), np.sum(x ** 2)]])
    b1 = np.array([np.sum(y), np.sum(x * y)])

    # Решаване на линейната система
    s1 = solve(A1, b1)
    P1 = lambda t: s1[0] + s1[1] * t

    # Изчисляване на абсолютната грешка
    absolute_error_1 = np.sqrt(np.sum((P1(x) - y) ** 2))

    return P1, absolute_error_1


# Полиномоализация от втора степен
def second_degree_polynomial(x, y):
    n = len(x)
    A2 = np.array([[n, np.sum(x), np.sum(x ** 2)],
                   [np.sum(x), np.sum(x ** 2), np.sum(x ** 3)],
                   [np.sum(x ** 2), np.sum(x ** 3), np.sum(x ** 4)]])
    b2 = np.array([np.sum(y), np.sum(x * y), np.sum((x ** 2) * y)])

    # Решаване на линейната система
    s2 = solve(A2, b2)
    P2 = lambda t: s2[0] + s2[1] * t + s2[2] * t ** 2

    # Изчисляване на абсолютната грешка
    absolute_error_2 = np.sqrt(np.sum((P2(x) - y) ** 2))

    return P2, absolute_error_2


# Графично изобразяване на резултатите
def plot_results(x, y, P1, P2, x_start, x_end):
    plt.scatter(x, y, color='red', label='Точки от данни')

    t_vals = np.linspace(x_start, x_end, 500)

    # Изобразяване на полином от първа степен
    plt.plot(t_vals, P1(t_vals), label='Полином от 1-ва степен', color='blue')

    # Изобразяване на полином от втора степен
    plt.plot(t_vals, P2(t_vals), label='Полином от 2-ра степен', color='green')

    plt.legend()
    plt.show()


def main():
    # Въвеждане на данни от потребителя за функцията и стъпката
    f, h, x_start, x_end = user_input()

    # Генериране на x и y стойности
    x, y = generate_data(f, h, x_start, x_end)

    # Полиномоализация от първа степен
    P1, absolute_error_1 = first_degree_polynomial(x, y)
    print("Полином от 1-ва степен: P1(t) = {:.4f} + {:.4f} * t".format(P1(0), P1(1) - P1(0)))
    print("Абсолютна грешка за полином от 1-ва степен: {:.4f}".format(absolute_error_1))

    # Полиномоализация от втора степен
    P2, absolute_error_2 = second_degree_polynomial(x, y)
    print("Полином от 2-ра степен: P2(t) = {:.4f} + {:.4f} * t + {:.4f} * t^2".format(P2(0), P2(1) - P2(0),
                                                                                        (P2(2) - P2(1)) / (1 - 0) ** 2))
    print("Абсолютна грешка за полином от 2-ра степен: {:.4f}".format(absolute_error_2))

    # Графично изобразяване на резултатите
    plot_results(x, y, P1, P2, x_start, x_end)


if __name__ == "__main__":
    main()
