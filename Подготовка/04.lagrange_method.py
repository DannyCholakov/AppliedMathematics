import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Функция за безопасно изчисление на входното изражение
def safe_eval(expr, t):
    return eval(expr)

# Входна функция f(t)
function_input = input("Въведете функцията f(t) (използвайте 't' като променлива, напр. 'math.sqrt(t)', 't**(1/3)', '1/(t**2 + 3)'): ")

# Дефиниране на функция за апроксимация
def f(t):
    return safe_eval(function_input, t)

# Въвеждане на стойностите на xx и точката (u)
xx = list(map(float, input("Въведете стойностите на xx, разделени с интервали: ").split()))
u = float(input("Въведете точката (u), в която искате да апроксимате стойността: "))
n = len(xx)
yy = [f(xx[i]) for i in range(n)]

# Дефиниране на Лагранжовия интерполационен полином
def L(t):
    result = 0
    for i in range(n):
        term = yy[i]
        for j in range(n):
            if i != j:
                term *= (t - xx[j]) / (xx[i] - xx[j])
        result += term
    return result

# Графика
t_values = np.linspace(min(xx) - 1, max(xx) + 1, 400)  # Диапазон, базиран на стойностите на xx
L_values = [L(t) for t in t_values]
plt.scatter(xx, yy, color='blue', label='Точки на данни')
plt.plot(t_values, L_values, color='red', label='L(t)', linewidth=2)
plt.title('Лагранжова интерполация')
plt.xlabel('Стойности на xx')
plt.ylabel('Стойности на yy')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()

# Извеждаме стойностите на xx и yy
print("Стойности на xx:", xx)
print("Стойности на yy:", yy)

# Символни изчисления
t_sym = sp.symbols('t')
function_input = function_input.replace("math.", "sp.")

try:
    f_sym = eval(function_input.replace("t", "t_sym"))
except Exception as e:
    print("Грешка при изчисляване на функцията:", e)
    exit(1)

# Изчисляване на Лагранжовия полином L(t)
L_sym = sum(yy[i] * sp.prod((t_sym - xx[j]) / (xx[i] - xx[j]) for j in range(n) if i != j) for i in range(n))

# Извеждаме L(t)
print(f"L(t) = {sp.expand(L_sym)}")

# Изчисляваме L(u) и точната стойност
approximation_at_u = L(u)
exact_value_at_u = f(u)
print(f"L({u}) = {approximation_at_u:.6f}, точна стойност = {exact_value_at_u:.6f}")

# Изчисляваме n-то производно на f(t)
n_derivative = sp.diff(f_sym, t_sym, n)

# Намиране на критични точки на производното (не е необходимо за този конкретен случай)
critical_points = sp.solve(n_derivative, t_sym)
# Можем да игнорираме критичните точки, тъй като не разглеждаме гранични стойности

# Оценяваме n-то производно в точките на xx
M = max([abs(n_derivative.evalf(subs={t_sym: xx[i]})) for i in range(n)], default=0)

# Извеждаме M
print(f"Mn+1 = {M:.5f}")

# Изчисляваме остатъка R
R = (M / math.factorial(n)) * np.prod([abs(u - xx[i]) for i in range(n)])
print(f"R = {abs(R):.10f}")
