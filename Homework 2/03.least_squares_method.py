import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# Даниел Чолаков ФК:2101681030  q=3 p=3

# Данни
x = np.array([0, 0.3, 0.63, 0.23, 0.35, 0.46, 0.71, 0.97])
y = np.array([2.01, 2.47, 2.80, 2.87, 3.17, 3.44, 4.56, 3.33])
z = 0.101 * 3

# Модел на полином от първа степен (линейна регресия)
A1 = np.vstack([np.ones(len(x)), x]).T
coef1, residuals1, _, _ = lstsq(A1, y, rcond=None)
P1 = lambda t: coef1[0] + coef1[1] * t
error1 = np.sqrt(np.sum((P1(x) - y) ** 2))

# Прогноза с помощта на модела от първа степен при z
P1_z = P1(z)

# Модел на полином от втора степен (квадратична апроксимация)
A2 = np.vstack([np.ones(len(x)), x, x**2]).T
coef2, residuals2, _, _ = lstsq(A2, y, rcond=None)
P2 = lambda t: coef2[0] + coef2[1] * t + coef2[2] * t**2
error2 = np.sqrt(np.sum((P2(x) - y) ** 2))

# Прогноза с помощта на модела от втора степен при z
P2_z = P2(z)

# Графика
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Точки на данните")

# Графика на модела от първа степен
x_plot = np.linspace(0, 1, 100)
plt.plot(x_plot, P1(x_plot), label=f"Модел от първа степен: y = {coef1[0]:.2f} + {coef1[1]:.2f} * x", color="red")

# Графика на модела от втора степен
plt.plot(x_plot, P2(x_plot), label=f"Модел от втора степен: y = {coef2[0]:.2f} + {coef2[1]:.2f} * x + {coef2[2]:.2f} * x^2", color="green")

plt.legend()
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Приближение по метода на най-малките квадрати")
plt.grid(True)
plt.show()

# Извеждане на резултати
print("Модел от първа степен:")
print(f"Уравнение: P1(t) = {coef1[0]:.2f} + {coef1[1]:.2f} * t")
print(f"Абсолютна грешка: {error1:.2f}")
print(f"Приближение при z = 0.101 * 3: P1(z) = {P1_z:.2f}")

print("\nМодел от втора степен:")
print(f"Уравнение: P2(t) = {coef2[0]:.2f} + {coef2[1]:.2f} * t + {coef2[2]:.2f} * t^2")
print(f"Абсолютна грешка: {error2:.2f}")
print(f"Приближение при z = 0.101 * 3: P2(z) = {P2_z:.2f}")


# Модел от първа степен:
# Уравнение: P1(t) = 2.33 + 1.65 * t
# Абсолютна грешка: 1.50
# Приближение при z = 0.101 * 3: P1(z) = 2.83
#
# Модел от втора степен:
# Уравнение: P2(t) = 1.91 + 4.23 * t + -2.63 * t^2
# Абсолютна грешка: 1.34
# Приближение при z = 0.101 * 3: P2(z) = 2.95
