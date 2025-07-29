import numpy as np

# Exercise with numpy arrays and matmul

X = np.array([
    [1, 2, -1],
    [1, 0, 1]
])

Y = np.array([
    [3, 1],
    [0, -1],
    [-2, 3]
])

Z = np.array([
    [1],
    [4],
    [6]
])

#Z = np.array([1, 4, 6])

A = np.array([
    [1, 2],
    [3, 5]
])

b = np.array([
    [5],
    [13]
])

print(X @ Y)
print(Y @ X)

Z_t = Z.transpose()
print(Z_t @ Y)

A_inv = np.linalg.inv(A)
print(A_inv @ b)