import sympy as sp

# Дефинираме символа x
x = sp.symbols('x')

# Въведена функция от потребителя
function_input = "3*x**5 - 3*x - cos(x + 5)"

# Използваме sympify с правилен контекст
f = sp.sympify(function_input, locals={"cos": sp.cos})

# Изчисляваме първата и втората производна
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

# Извеждаме резултатите
print(f"Функцията f(x): {f}")
print(f"Първа производна f'(x): {f_prime}")
print(f"Втора производна f''(x): {f_double_prime}")
