num1 = 0
num2 = 1
n = 15
print(num1, end=" ")
print(num2, end=" ")
while(n - 2):
    n -= 1
    sum1 = num1 + num2
    num1 = num2
    num2 = sum1
    print(sum1, end=" ")