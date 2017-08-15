from plot_signal import plot_signal
import numpy as np
import matplotlib.pyplot as plt

with open("unlabelled/1_2967.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip()) for x in content]
x, y, z = range(4000), np.array(content[0:4000]), np.ones(4000)
plot_signal(x, y, z)
plt.show()
