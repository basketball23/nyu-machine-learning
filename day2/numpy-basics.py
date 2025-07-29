import numpy as np
import random

u = [1, 2, 3, 4, 5]
v = [6, 7, 8, 9, 10]

# Numpy arrays

x = np.array(u)
y = np.array(v)

(x)
(y)

# Numpy array operations, term by term

(x + y)
(x - y)
(x * y)
(x / y)
(x ** y)

# Dot product

(np.dot(x, y))
(x.dot(y))
(x @ y)

# Norm of a vector can be calculated by dot product of the vector with itself, then square rooting
(np.sqrt(x @ x))
# Or by using the norm function, under linalg
(np.linalg.norm(x))

#---------------------------------------------------------#

## Matrices

M = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

(M.shape)

# Transposition

(M.transpose())
(M.T)

# Inverse

M_inv = np.linalg.inv(M)

# Matrix multiplication, should return identity matrix
np.matmul(M, M_inv)
M @ M_inv


## Matrix-Vector operations

A = np.array([
    [2, 0],
    [0, 1],
    [1, 1]
])

# Row vector
u = np.array([[1, 2]])

# Column vector
u = np.array([[1], [2]])

# 1d array, shape will print as (2, )
u = np.array([1, 2])

print(f"A shape: {A.shape}")
print(f"u shape: {u.shape}")

result = A @ u
print(result)
print(f"A@u shape: {result.shape}")