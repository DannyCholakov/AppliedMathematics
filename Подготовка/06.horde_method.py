import sympy as sp
import math


def find_interval(f, start=-10, end=10, step=0.1):
    """
    Намира интервал [a, b], в който има корен на f(x).
    Претърсва диапазона [start, end] със стъпка step.
    """
    x = sp.symbols('x')
    f_numeric = sp.lambdify(x, f, 'math')

    for i in range(int((end - start) / step)):
        a = start + i * step
        b = a + step
        # Проверка за смяна на знака
        if f_numeric(a) * f_numeric(b) < 0:
            return a, b
    raise ValueError("Не е намерен интервал със смяна на знака в дадения диапазон.")


def chord_method(f, a, b, epsilon=1e-4):
    """
    Намира корен на уравнението f(x) = 0 с метода на хордите.
    """
    x = sp.symbols('x')
    f_derivative_1 = sp.diff(f, x)
    f_derivative_2 = sp.diff(f_derivative_1, x)

    # Проверка на условията за прилагане
    print("\nПроизводни на функцията:")
    print("Първа производна на f(x):", f_derivative_1)
    print("Втора производна на f(x):", f_derivative_2)

    # Функцията за итерации
    f_numeric = sp.lambdify(x, f, 'math')

    def next_x(x0, x1):
        return x1 - f_numeric(x1) * (x1 - x0) / (f_numeric(x1) - f_numeric(x0))

    # Начални стойности
    x0, x1 = a, b
    iterations = []
    for i in range(100):  # Ограничен брой на итерациите
        x2 = next_x(x0, x1)
        error = abs(x2 - x1)
        iterations.append((i + 1, x2, error))
        if error < epsilon:
            break
        x0, x1 = x1, x2

    return iterations


if __name__ == "__main__":
    # Вход от потребителя
    print("Метода на Хордите / Даниел Чолаков")
    print("Въведи уравнението.")
    function_input = input("Уравнение f(x): ").strip()
    epsilon = float(input("Точност epsilon (напр. 0.0001): "))

    # Преобразуване на уравнението
    x = sp.symbols('x')
    f = sp.sympify(function_input)

    # Автоматично намиране на интервала
    print("\nТърсене на подходящ интервал...")
    try:
        a, b = find_interval(f)
        print(f"Намерен интервал: [a, b] = [{a}, {b}]")
    except ValueError as e:
        print(str(e))
        exit()

    # Решаване с метода на хордите
    iterations = chord_method(f, a, b, epsilon=epsilon)

    # Отпечатване на резултатите
    print("\nРезултати от метода:")
    print(f"{'Итерация':<10}{'Приближение x':<20}{'Грешка':<10}")
    for i, x_val, error in iterations[:3]:  # Първите 3 итерации
        print(f"{i:<10}{x_val:<20.10f}{error:<10.10f}")

    print("\nКраен резултат:")
    final_iteration = iterations[-1]
    print(f"x = {final_iteration[1]:.10f} с грешка {final_iteration[2]:.10f}")
