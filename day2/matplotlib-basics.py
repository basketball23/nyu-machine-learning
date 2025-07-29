import matplotlib.pyplot as plt
import numpy as np

# Initialize x and y coordinate arrays as numpy arrays
x = np.array(([0, 1, 2]))
y = np.array([2, 5, 8])

# Image resize
plt.figure(figsize=(4, 4))

# Display points
plt.scatter(x, y, color="#E6E6FA")

# Display line
plt.plot(x, y, color="#E4F33DFF")

# Display grid
plt.grid(True)

plt.show()