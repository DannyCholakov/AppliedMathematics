from itertools import combinations
from scipy.stats import binom

# Стойност
p = 3

# Вероятности
p1 = 0.8 + p / 100
p2 = 0.75 + p / 100
p3 = 0.6 + p / 100
probabilities = [p1, p2, p3]

# Брой опити (3-ма стрелци)
n = len(probabilities)

# Функция за вероятност на точно k улучвания
def probability_exact_hits(k, probabilities):
    combinations_prob = 0
    for hit_indices in combinations(range(len(probabilities)), k):
        prob = 1
        for i in range(len(probabilities)):
            if i in hit_indices:
                prob *= probabilities[i]
            else:
                prob *= (1 - probabilities[i])
        combinations_prob += prob
    return combinations_prob

# a) Вероятност за най-много 1 улучване
prob_at_most_1 = sum(probability_exact_hits(k, probabilities) for k in range(0, 2))
print(f"Вероятност целта да е поразена най-много веднъж: {prob_at_most_1:.4f}")

# б) Вероятност за 1 или 2 улучвания
prob_1_or_2 = sum(probability_exact_hits(k, probabilities) for k in range(1, 3))
print(f"Вероятност целта да е поразена един или два пъти: {prob_1_or_2:.4f}")
