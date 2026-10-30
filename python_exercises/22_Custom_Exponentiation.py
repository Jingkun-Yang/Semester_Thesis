exp = 3
base = 3
def exponent(base, exp):
    result = 1
    for i in range(0, exp):
        result *= base
    return result
print(f"{base} raises to the power of {exp}: {exponent(base, exp)}")
