from numpy import array
from numpy import linalg as LA
from numpy import inf
import numpy as np

"""
%Gauss-Seidel   Use both the Jacobi method and the Gauss-Seidel method to solve the indicated linear system of equations.
%             
%             
%
%     inputs:
%             A       coefficient matrix for linear system
%                     (matrix must be square)
%             b       right-hand side vector
%
%     output:
%             x       solution vector (i.e., vector for which Ax = b)
%
%
"""

ITERATION_LIMIT = 1000

# initialize the matrix
A = array([
    [4.0, -1, 0, -2, 0, 0],
    [-1, 4, -1, 0, -2, 0], 
    [0, -1, 4, 0, 0, -2], 
    [-1, 0, 0, 4, -1, 0], 
    [0, -1, 0, -1, 4, -1], 
    [0, 0, -1, 0, -1, 4]])
# initialize the b vector
b = array([-1.0,0,1,-2,1,2])

n  = 0  #iteration counter

# prints the equations of the linear system (input)
print("System of equations:")
for i in range(A.shape[0]):
    row = ["{0:3g}*x{1}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

# initial vector
x = np.zeros_like(b)

# Gauss-Seidel method
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

    # Number of iterations for Gauss-Seidel method
    xn = LA.norm(x-it_count,inf)
    n = n + 1

# Output 
print("approximate x = {0}".format(x))
error = np.dot(A, x) - b
print("Error: {0}".format(error))
print(f" Number of iterations for Gauss-Seidel = {n}") 