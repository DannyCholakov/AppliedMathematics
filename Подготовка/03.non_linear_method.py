import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    # Въвеждаме стойности за x и y от потребителя
    x_values = input("Въведете стойности за x, разделени с интервали: ")
    y_values = input("Въведете стойности за y, разделени с интервали: ")

    # Преобразуваме въведените стойности в numpy масиви
    x = np.array([float(val) for val in x_values.split()])
    y = np.array([float(val) for val in y_values.split()])

    # Проверяваме дали броят на стойностите на x и y е еднакъв
    if len(x) != len(y):
        raise ValueError("Броят на стойностите за x и y трябва да е същият.")

    # Въвеждаме функцията, която ще бъде аппроксимирана
    user_function = input("Въведете функцията за апроксимация (използвайте 'a0', 'a1', 'a2' и т.н. за параметри и 'x' като променлива):\n")

    # Преобразуваме въведената функция в Python функция
    def func(x, *params):
        param_dict = {f'a{i}': params[i] for i in range(len(params))}
        return eval(user_function, {"np": np, "x": x}, param_dict)

    # Познаваме броя на параметрите (броим срещанията на 'a')
    num_params = user_function.count('a')
    if num_params == 0:
        raise ValueError("Не са открити параметри 'a0', 'a1' и т.н. във функцията.")

    # Извършваме аппроксимация на кривата
    popt, pcov = curve_fit(func, x, y, p0=[1] * num_params)

    # Извеждаме получената функция
    param_str = " + ".join([f"{popt[i]}*x^{i}" for i in range(len(popt))])
    fitted_model = f"Моделираната функция: {param_str}"
    print(fitted_model)

    # Изчертаваме оригиналните данни и моделираната крива
    plt.scatter(x, y, color="blue", label="Данни")
    t_vals = np.linspace(min(x), max(x), 100)
    plt.plot(t_vals, func(t_vals, *popt), label=fitted_model, color="orange")
    plt.legend()
    plt.show()

    # Изчисляваме грешката
    absolute_error = np.sum((func(x, *popt) - y) ** 2)
    print(f"Абсолютна грешка: {absolute_error}")

except ValueError as ve:
    print(f"Грешка в стойността: {ve}")
except SyntaxError:
    print("Синтактична грешка: Имаш проблем със синтаксиса на функцията.")
except NameError as ne:
    print(f"Грешка в името: {ne}")
except Exception as e:
    print(f"Неочаквана грешка: {e}")
