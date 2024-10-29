import numpy as np
import matplotlib.pyplot as plt

# Даниел Чолаков ФК:2101681030  q=3 p=3

# Дефиниране на функцията f(x) и нейните първа и втора производни
def f(x):
    return np.sin(7 * x + np.sqrt(7 + x**2)) - 7 * x + np.log(2 + x) + 4

def df(x):
    return (7 * np.cos(7 * x + np.sqrt(7 + x**2)) +
            (x / np.sqrt(7 + x**2)) * np.cos(7 * x + np.sqrt(7 + x**2)) -
            7 + 1 / (2 + x))

def ddf(x):
    # Численна производна на df за простота
    h = 1e-5
    return (df(x + h) - df(x - h)) / (2 * h)

# Графика на функцията f(x)
x_values = np.linspace(0.5, 1, 400)
y_values = f(x_values)
plt.plot(x_values, y_values, label="f(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Графика на f(x)")
plt.legend()
plt.show()

# Графика на първата производна f'(x)
y_prime_values = df(x_values)
plt.plot(x_values, y_prime_values, label="f'(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Графика на f'(x)")
plt.legend()
plt.show()

# Графика на втората производна f''(x)
y_double_prime_values = ddf(x_values)
plt.plot(x_values, y_double_prime_values, label="f''(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f''(x)")
plt.title("Графика на f''(x)")
plt.legend()
plt.show()

# Метод на Нютон за намиране на корен
eps = 0.0001  # толеранс
x0 = 1  # начална стойност
max_iter = 100

print("Начално приложение")
for n in range(max_iter):
    f_x0 = f(x0)
    df_x0 = df(x0)

    # Проверка за невалидни стойности
    if np.isnan(f_x0) or np.isnan(df_x0):
        print("Намерена е невалидна стойност по време на изчислението. Изход.")
        break

    # Проверка дали производната е нула
    if df_x0 == 0:
        print("Производната е нула. Няма решение.")
        break

    # Стъпка на Нютон
    x1 = x0 - f_x0 / df_x0
    err = abs(x1 - x0)
    print(f"m = {n}, x = {x0:.10f}, f(x0) = {f_x0:.10f}, грешка = {err:.10f}")

    # Проверка дали грешката е под толеранса
    if err < eps:
        print(f"Отговор х = {x1:.10f} с грешка = {err:.10f}")
        break

    x0 = x1
else:
    print("Достигнат е максималният брой итерации. Решението може да не е сближено.")

# Начално приложение
# m = 0, x = 1.0000000000, f(x0) = -2.2941645560, грешка = 0.1708334978
# m = 1, x = 0.8291665022, f(x0) = -0.0142408131, грешка = 0.0012410420
# m = 2, x = 0.8279254603, f(x0) = -0.0000311119, грешка = 0.0000027232
# Отговор х = 0.8279227371 с грешка = 0.0000027232