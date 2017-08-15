import numpy as np
import matplotlib.pyplot as plt

def get_heatmap_values(x):
    y = [n[0] - n[1] for n in x]
    normalize = [(n - (-1)) / (1 - (-1)) for n in y]
    return normalize

x = np.array([[0.2, 0.9], [0.9, 0.2]])
mean = np.mean(x, axis=1)
print(x - mean)
