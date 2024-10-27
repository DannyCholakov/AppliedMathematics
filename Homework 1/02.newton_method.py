import numpy as np

# Дадени стойности за p и q
p = 3
q = 3
tolerance = 1e-5
max_iterations = 100

# Част (а): Дефиниране на нелинейната функция и нейната производна
print("Част (а): Дефиниране на функцията и производната")
def f(x):
    if x <= -2:
        return float('inf')  # Връща голяма стойност, ако x не е в допустимия домейн
    return np.sin(p * x + np.sqrt(q + x**2)) - p * x + np.log(2 + x) + 4

def f_prime(x):
    if x <= -2:
        return float('inf')  # Връща голяма стойност, ако x не е в допустимия домейн
    return (np.cos(p * x + np.sqrt(q + x**2)) * (p + x / np.sqrt(q + x**2))) - p + 1 / (2 + x)

# Част (б): Метод на Нютон
print("Част (б): Метод на Нютон за намиране на корен")
def newton_method(x0, tolerance, max_iterations):
    for i in range(max_iterations):
        fx = f(x0)
        fpx = f_prime(x0)
        # Проверка за допустимост на fx и fpx
        if abs(fx) == float('inf') or fpx == float('inf'):
            print("Невалидна стойност на x в домейна на логаритъма. Опитайте с различно начално предположение.")
            return None
        if abs(fx) < tolerance:
            print(f"Намерен корен: x = {x0:.5f} след {i+1} итерации")
            return x0
        if fpx == 0:
            print("Производната е нула, методът на Нютон не може да продължи.")
            return None
        x0 = x0 - fx / fpx
        # Допълнителна проверка за граници на x
        if x0 <= -2:
            print("Прескочено в невалиден домейн за логаритъма. Опитайте различно начално предположение.")
            return None
    print("Максималният брой итерации е достигнат без намиране на решение.")
    return None

# Опит с различни начални предположения
initial_guesses = [1.0, 0.5, 2.0, -1.5]  # Списък с начални предположения
for guess in initial_guesses:
    print(f"\nОпит с начално предположение x = {guess}")
    root = newton_method(guess, tolerance, max_iterations)
    print("Резултат за корена:", root)
