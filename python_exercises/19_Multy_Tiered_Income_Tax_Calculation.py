income = 45000.
Tax = 0.

level1 = 10000.
level2 = 10000.


if income >= level1:
    Tax += 0
    income -= level1
    if income >= level2:
        Tax += level2 * 0.1
        income -= level2
        Tax += income * 0.2
    else:
        Tax += income * 0.1
else:
    Tax = 0

print(Tax)