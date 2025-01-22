import numpy as np
import matplotlib.pyplot as plt

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

    # Дефинираме линейния модел
    A = np.vstack([x, np.ones(len(x))]).T
    a0, a1 = np.linalg.lstsq(A, y, rcond=None)[0]

    # Извеждаме моделираното уравнение
    fitted_model = f"Моделираната функция: {a0} + {a1}x"
    print(fitted_model)

    # Изчертаваме оригиналните данни и получената линейна линия
    plt.scatter(x, y, color="blue", label="Данни")
    plt.plot(x, a0 + a1 * x, label=f"{a0:.2f} + {a1:.2f}x", color="orange")
    plt.legend()
    plt.show()

    # Изчисляваме грешката
    absolute_error = np.sum((a0 + a1 * x - y) ** 2)
    print(f"Абсолютна грешка: {absolute_error}")

except ValueError as ve:
    print(f"Грешка в стойността: {ve}")
except Exception as e:
    print(f"Неочаквана грешка: {e}")
