from numpy import array
from numpy import linalg as LA
from numpy import inf
import numpy as np

"""
%Gauss-Seidel   Use both the Jacobi method and the Gauss-Seidel method to solve the indicated linear system of equations.
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
    [4, -1, 0, -2, 0, 0],
    [-1, 4, -1, 0, -2, 0], 
    [0, -1, 4, 0, 0, -2], 
    [-1, 0, 0, 4, -1, 0], 
    [0, -1, 0, -1, 4, -1], 
    [0, 0, -1, 0, -1, 4]])
# initialize the b vector
b = array([-1.0,0,1,-2,1,2])

# assigning some variables that will be used in the functions
n  = 0  #iteration counter
tol = 5e-6 # iterations should terminate when my iterantions falls bellow 5 X 10^-6
converged = False # for convergence test


# testing if A is diagonally dominant
diag = np.diag(np.abs(A)) # Find diagonal coefficients

off_diag = np.sum(np.abs(A), axis=1) - diag # Find row sum without diagonal

if np.all(diag >= off_diag): #printing the results of the calculation : if the sum of the other coefficients is smaller or larger than the highest magnitude of the diagonal coefficients the system will give us these 2 outputs
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')


# prints the equations of the linear system A(input)
print("System of equations:")
for i in range(A.shape[0]):
    row = ["{0:3g}*x{1}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

# initial vector
x = np.zeros_like(b)

# Gauss-Seidel method
for i in range(ITERATION_LIMIT):
    x_new = np.zeros_like(x) # using initial vector

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
    
    dx = np.sqrt(np.dot(x_new-it_count, x_new-it_count))
    if dx < tol:
        converged = True
        print('Converged!')
        break

if not converged:
    print('Not converge, increase the # of iterations')

# Output 
print("approximate x = {0}".format(x))
error = np.dot(A, x) - b
print("Error: {0}".format(error))
print(f" Number of iterations for Gauss-Seidel = {n}") 