from collections import Counter


words = [
    "luxuriant", "silly", "dizzy", "frightening", "blink", "silly", "enjoy",
    "suspend", "blink", "reward", "blink", "fact", "debt", "marble", "blink",
     "yak", "frightening", "suspend", "debt"]

word_count = Counter(words)

for word, count in word_count.items():
    print(f"{word} |{'*' * count}")