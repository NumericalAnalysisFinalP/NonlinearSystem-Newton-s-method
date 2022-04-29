from numpy import array
from numpy import linalg as LA
from numpy import inf
import numpy as np


ITERATION_LIMIT = 1000

# initialize the matrix
A = array([
    [4.0, -1, 0, -2, 0, 0],
    [-1, 4, -1, 0, -2, 0], 
    [0, -1, 4, 0, 0, -2], 
    [-1, 0, 0, 4, -1, 0], 
    [0, -1, 0, -1, 4, -1], 
    [0, 0, -1, 0, -1, 4]])
# initialize the RHS vector
b = array([-1.0,0,1,-2,1,2])

n  = 0  #iteration counter

print("System of equations:")
for i in range(A.shape[0]):
    row = ["{0:3g}*x{1}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

x = np.zeros_like(b)
for it_count in range(1, ITERATION_LIMIT):
    x_new = np.zeros_like(x)
    print("Iteration {0}: {1}".format(it_count, x))
    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x_new[:i])
        s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
    if np.allclose(x, x_new, rtol=5e-6):
        break
    x = x_new

    # Number of iterations for Jacobi
    xn = LA.norm(x-it_count,inf)
    n = n + 1

print("Solution: {0}".format(x))
error = np.dot(A, x) - b
print("Error: {0}".format(error))
print(f" Number of iterations for Gauss-Seidel = {n}") 