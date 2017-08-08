import numpy as np

def get_heatmap_values(x):
    y = [n[0] - n[1] for n in x]
    normalize = [(n - (-1)) / (1 - (-1)) for n in y]
    return normalize

x = np.array([[0.2, 0.9], [0.9, 0.2]])
print(get_heatmap_values(x))
