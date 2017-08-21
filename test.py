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

data_formatter = Data_Formatter()
data_formatter.assign_data(x, y)
print(data_formatter.x)
print(data_formatter.y)
data_formatter.split_training_testing(0.1)
print("\n")
print(data_formatter.x_train)
print(data_formatter.y_train)
print("\n")
print(data_formatter.x_test)
print(data_formatter.y_test)
