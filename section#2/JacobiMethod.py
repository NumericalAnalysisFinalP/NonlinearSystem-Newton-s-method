import numpy as np
from numpy import array
from numpy import linalg as LA
from numpy import inf

ITERATION_LIMIT = 1000

# initialize the matrix
A = array(
    [[4.0, -1, 0, -2, 0, 0], 
    [-1, 4, -1, 0, -2, 0], 
    [0, -1, 4, 0, 0, -2], 
    [-1, 0, 0, 4, -1, 0], 
    [0, -1, 0, -1, 4, -1], 
    [0, 0, -1, 0, -1, 4]])
b = array([-1.0,0,1,-2,1,2])


n = 0
# prints the system
print("System:")
for i in range(A.shape[0]):
    row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print(f'{" + ".join(row)} = {b[i]}')
print()

x = np.zeros_like(b)
for it_count in range(ITERATION_LIMIT):
    if it_count != 0:
        print("Iteration {0}: {1}".format(it_count, x))
    x_new = np.zeros_like(x)

    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if x_new[i] == x_new[i-1]:
          break

    if np.allclose(x, x_new, atol=5e-6, rtol=0.):
        break

    x = x_new

    # Number of iterations for Jacobi

    xn = LA.norm(x-it_count,inf)
    n = n + 1

print("Solution: ")
print(x)
error = np.dot(A, x) - b
print("Error:")
print(error)
print(f" Number of iterations for Jacobi = {n}")      
