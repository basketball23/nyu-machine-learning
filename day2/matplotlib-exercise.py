import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 5, 3, 4])
y = np.array([1, 2, 3, 1])

plt.scatter(x, y)
plt.plot(x, y, color="red")
plt.show()
