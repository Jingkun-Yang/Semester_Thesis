num = 1234321
ori = num
rev = 0
while(num):
    rev *= 10
    rev += num % 10
    num //= 10
print(rev)
if rev == ori:
    print(f"{ori} is a palindrome number.")

str_ori = str(ori)
if str_ori == str_ori[::-1]:
    print(f"str {str_ori} is a palindrome number.")