def remove_chars(word, n):
    res = word[n:]
    return res

word = 'pynative'

print(remove_chars(word, 2))
print(remove_chars(word, 4))