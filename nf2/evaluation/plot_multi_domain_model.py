import matplotlib.pyplot as plt
import numpy as np

Mm_per_ds = 100
overlap_width = 0.5
model_config = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50],
                [50, 60], [60, 70], [70, 80], [80, 90], [90, 100]]
z_range = np.linspace(0, 100, 1000)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

windows = []
for config in model_config:
    start = config[0] / Mm_per_ds
    end = config[1] / Mm_per_ds
    overlap = overlap_width / Mm_per_ds
    z = z_range / Mm_per_ds
    #
    left = sigmoid((z - start) * 2 / overlap) if start > 0 else 1
    right = sigmoid((z - end) * 2 / overlap) if end < 1 else 0
    window = left - right
    #
    windows.append(window)

for i, window in enumerate(windows):
    plt.plot(window, z_range, label=f'window {i} - {model_config[i]}')

plt.legend()
plt.xlabel('window')
plt.ylabel('z [Mm]')
plt.show()