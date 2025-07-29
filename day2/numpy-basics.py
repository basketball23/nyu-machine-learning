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

# Matrices #

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

np.matmul(M, M_inv)
M @ M_inv