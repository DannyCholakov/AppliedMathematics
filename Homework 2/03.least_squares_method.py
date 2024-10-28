import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

# Дадени стойности за p и q
p = 3
q = 3

# Данни
t_values = np.array([0, 0.1 * p, 0.21 * q, 0.23, 0.35, 0.46, 0.71, 0.97])
f_values = np.array([2.01, 2.47, 2.80, 2.87, 3.17, 3.44, 3.56, 3.33])

# Част (а): Линеен модел чрез метода на най-малките квадрати
print("Част (а): Линеен модел чрез метода на най-малките квадрати")
linear_model = np.polyfit(t_values, f_values, 1)
print(f"Линеен модел (степен 1): y = {linear_model[0]:.4f} * x + {linear_model[1]:.4f}")

# Част (б): Модел от втора степен чрез метода на най-малките квадрати
print("Част (б): Квадратичен модел чрез метода на най-малките квадрати")
quadratic_model = np.polyfit(t_values, f_values, 2)
print(f"Квадратичен модел (степен 2): y = {quadratic_model[0]:.4f} * x^2 + {quadratic_model[1]:.4f} * x + {quadratic_model[2]:.4f}")

# Част (в): Изчисляване на грешката за всеки модел
print("Част (в): Изчисляване на грешката за всеки модел")
linear_predictions = np.polyval(linear_model, t_values)
quadratic_predictions = np.polyval(quadratic_model, t_values)

linear_error = np.mean((f_values - linear_predictions)**2)
quadratic_error = np.mean((f_values - quadratic_predictions)**2)

print(f"Грешка на линейния модел: {linear_error:.4f}")
print(f"Грешка на квадратичния модел: {quadratic_error:.4f}")

# Част (г): Приближени стойности за двата модела при z = 0.101 * p
z = 0.101 * p
print(f"Част (г): Приближени стойности за двата модела при z = {z:.4f}")
linear_estimate = np.polyval(linear_model, z)
quadratic_estimate = np.polyval(quadratic_model, z)

print(f"Линеен модел при z = {z:.4f}: {linear_estimate:.4f}")
print(f"Квадратичен модел при z = {z:.4f}: {quadratic_estimate:.4f}")
