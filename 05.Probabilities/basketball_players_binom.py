import math
import scipy.stats as stats

# Дадени параметри
p = 0.82
n = 10

# Математическо очакване, дисперсия и стандартно отклонение
E_xi = n * p
Var_xi = n * p * (1 - p)
Std_xi = math.sqrt(Var_xi)

# a) Закон за разпределение на случайна величина ξ
# Това ще бъде биномно разпределение
xi_distribution = [stats.binom.pmf(k, n, p) for k in range(n+1)]

# b) Вероятността P(ξ ≥ 1)
P_xi_geq_1 = 1 - stats.binom.pmf(0, n, p)

# c) Вероятността P(2 ≤ ξ ≤ 4)
P_2_to_4 = stats.binom.cdf(4, n, p) - stats.binom.cdf(1, n, p)

# d) Най-вероятният брой попадения (на практика, максимум на P(ξ = k))
most_probable = max(range(n+1), key=lambda k: stats.binom.pmf(k, n, p))

# Извеждаме резултатите
print(f"Математическо очакване (E(ξ)): {E_xi:.4f}")
print(f"Дисперсия (Var(ξ)): {Var_xi:.4f}")
print(f"Стандартно отклонение (σ(ξ)): {Std_xi:.4f}")
print(f"Вероятността P(ξ ≥ 1): {P_xi_geq_1:.4f}")
print(f"Вероятността P(2 ≤ ξ ≤ 4): {P_2_to_4:.4f}")
print(f"Най-вероятният брой попадения в коша: {most_probable}")
