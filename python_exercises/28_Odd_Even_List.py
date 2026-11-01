numbers = [12, 7, 34, 21, 5, 10, 8, 3, 19, 2]
even = []
odd = []
for n in numbers:
    if n % 2 == 0:
        even.append(n)
    else:
        odd.append(n)

print(f"Even numbers are: {even}")
print(f"Odd numbers are {odd}")