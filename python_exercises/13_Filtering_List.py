num_list = [10, 20, 33, 46, 55]

def divisible(num_list):
    num_divisible = []
    denominator = 5
    for num in num_list:
        if num % denominator == 0:
            num_divisible.append(num)
    return num_divisible

num_result = divisible(num_list)
print(num_result)