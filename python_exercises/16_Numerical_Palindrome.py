number1 = 121
number2 = 125

def palindrome(num):
    str_num = str(num) # only string can be inversed using [::-1]
    inverse_num = str_num[::-1]
    if str_num == inverse_num:
        return True
    else:
        return False

print(f"{number1} is palindrome is", palindrome(number1))
print(f"{number2} is palindrome is", palindrome(number2))