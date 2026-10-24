sentence = "Learning Python is fun!"

vowels = "aeiou"

number = 0.0

for i in sentence.lower():
    if i in vowels:
        number += 1

print(number)