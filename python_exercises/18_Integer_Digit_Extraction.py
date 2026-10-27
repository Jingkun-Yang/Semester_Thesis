int1 = int(7536987640)
reversal1 = 0
while(int1):
    reversal1 *= 10
    reversal1 += int1 % 10
    print(int1 % 10)
    int1 //= 10

print(reversal1)