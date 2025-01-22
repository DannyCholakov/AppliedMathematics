import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Функция за метода на Нютон
def newton_method(f, f_prime, x0, epsilon=1e-4, max_iter=100):
    """
    Намира корен на уравнението f(x) = 0 с метода на Нютон.
    """
    x = sp.symbols('x')

    # Ламбда функции за числено изчисляване
    f_numeric = sp.lambdify(x, f, 'math')
    f_prime_numeric = sp.lambdify(x, f_prime, 'math')

    iterations = []

    for i in range(max_iter):
        fx = f_numeric(x0)
        fx_prime = f_prime_numeric(x0)

        if abs(fx_prime) < 1e-10:
            print("Проблем: Първата производна е твърде малка.")
            break

        # Итерация на метода на Нютон
        x_new = x0 - fx / fx_prime
        error = abs(x_new - x0)

        iterations.append((i + 1, x_new, error))

        if error < epsilon:
            break

        x0 = x_new

    return iterations


# Основна функция за решаване
def solve_newton_method(function_input, p_val, q_val, epsilon=1e-4, x0=0):
    # Дефинираме променливите
    x, p, q = sp.symbols('x p q')

    # Преобразуваме въведеното уравнение в символичен израз
    f = sp.sympify(function_input)

    # Изчисляваме първата и втората производна на функцията
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)

    # Подставяме стойности за p и q
    f_numeric = f.subs({p: p_val, q: q_val})
    f_prime_numeric = f_prime.subs({p: p_val, q: q_val})
    f_double_prime_numeric = f_double_prime.subs({p: p_val, q: q_val})

    print("\nФункция:", f_numeric)
    print("Първа производна:", f_prime_numeric)
    print("Втора производна:", f_double_prime_numeric)

    # Намираме корена с метода на Нютон
    iterations = newton_method(f, f_prime, x0, epsilon)

    # Отпечатваме резултатите
    print("\nРезултати от метода на Нютон:")
    print(f"{'Итерация':<10}{'Приближение x':<20}{'Грешка':<10}")
    for i, x_val, error in iterations[:3]:  # Първите 3 итерации
        print(f"{i:<10}{x_val:<20.10f}{error:<10.10f}")

    print("\nКраен резултат:")
    final_iteration = iterations[-1]
    print(f"x = {final_iteration[1]:.10f} с грешка {final_iteration[2]:.10f}")

    return iterations, f, x


# Функция за графиката
def plot_function_and_iterations(f, iterations, x0, epsilon=1e-4):
    # Преобразуваме функциите за числено изчисляване
    f_numeric = sp.lambdify('x', f, 'math')

    # Генерираме точки за графиката
    x_vals = np.linspace(-5, 5, 400)
    y_vals = f_numeric(x_vals)

    # Чертаем графиката
    plt.plot(x_vals, y_vals, label=str(f))
    plt.axhline(0, color='black', linewidth=1)  # Оста X
    plt.axvline(0, color='black', linewidth=1)  # Оста Y

    # Добавяме итерациите на метода на Нютон
    x_iter = [x0]  # Началната стойност
    for i, x_val, error in iterations:
        x_iter.append(x_val)

    y_iter = [f_numeric(x) for x in x_iter]

    # Чертаем точките на итерациите
    plt.scatter(x_iter, y_iter, color='red', label='Итерации', zorder=5)

    # Свързваме точките с линии
    plt.plot(x_iter, y_iter, 'r--', alpha=0.6)

    # Настройки на графиката
    plt.title("Метод на Нютон - Итерации")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    # Задаваме новия диапазон за осите
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.xticks(np.arange(-5, 5, 1))  # Деления на x ос с по-малка стъпка
    plt.yticks(np.arange(-5, 5, 1))  # Деления на y ос с по-малка стъпка

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Вход от потребителя
    print("Метод на Нютон")
    print("Въведи уравнението.")
    function_input = input("Уравнение f(x) (например: sin(x + p) + p*x**3 - 3*q): ").strip()
    epsilon = float(input("Точност epsilon (напр. 0.0001): "))
    x0 = float(input("Начално приближение x0 (напр. 0): "))
    p_val = float(input("Стойност на p: "))
    q_val = float(input("Стойност на q: "))

    # Решаване с метода на Нютон
    iterations, f, x = solve_newton_method(function_input, p_val, q_val, epsilon, x0)

    # Чертаем графика с итерациите
    plot_function_and_iterations(f, iterations, x0, epsilon)
