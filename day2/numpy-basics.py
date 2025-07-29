import numpy as np
import random

u = [1, 2, 3, 4, 5]
v = [6, 7, 8, 9, 10]

# Numpy arrays

x = np.array(u)
y = np.array(v)

print(x)
print(y)

# Numpy array operations, term by term

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)

# Dot product

print(np.dot(x, y))
print(x.dot(y))
print(x @ y)

# Norm of a vector can be calculated by dot product of the vector with itself, then square rooting
print(np.sqrt(x @ x))
# Or by using the norm function, under linalg
print(np.linalg.norm(x))

#---------------------------------------------------------#

# Matrices #

M = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

print(M.shape)