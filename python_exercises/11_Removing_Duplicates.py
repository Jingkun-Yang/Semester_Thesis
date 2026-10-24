data = [2, 2, 1, 2, 2, 3, 4, 4, 4, 5]
unique_data = list(set(data))

print(unique_data)
# set is designed to store only unique elements.
# converting to set loses the original order of items.

data = [2, 2, 1, 4, 4, 4, 5, 3, 3]
unique_data = dict.fromkeys(data)
print(unique_data)