import numpy as np
import matplotlib.pyplot as plt
from data_formatter import Data_Formatter

y = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])

x = np.array([
    [1],
    [2],
    [3],
    [4]
])


print(np.mean(x, axis=1))
