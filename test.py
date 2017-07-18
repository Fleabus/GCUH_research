import numpy as np

arr = np.array([1, 2, 1.5, 3])

mean = np.mean(arr)

new_arr = arr - mean
print(new_arr)
