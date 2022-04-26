from numpy import array
from numpy import linalg as LA
from numpy import inf
import numpy as np
import copy

# given linear system

A = array([
 [4.0, -1, 0, -2, 0, 0],
 [-1, 4, -1, 0, -2, 0], 
 [0, -1, 4, 0, 0, -2], 
 [-1, 0, 0, 4, -1, 0], 
 [0, -1, 0, -1, 4, -1], 
 [0, 0, -1, 0, -1, 4]])
b = array([-1.0,0,1,-2,1,2])

#first check if the coefficient matrix is diagonally dominant or not

# Find diagonal coefficients
diag = np.diag(np.abs(A)) 

# Find row sum without diagonal
off_diag = np.sum(np.abs(A), axis=1) - diag 

if np.all(diag >= off_diag):
    print('matrix is diagonally dominant')
else:
    print('NOT diagonally dominant')

#m = len(b) # return the array is the same length as b
x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
x6 = 0
epsilon = 5*10**-6 # iteration max limit
converged = False

x_old = np.array([x1, x2, x3, x4, x5, x6])

print('Iteration results')
print('k,    x1,     x2,     x3,     x4,      x5,      x6')
for k in range(1, 50):
    x1 = (-1+x2+2*x4)/(4)
    x2 = (x1+x3+2*x5)/(4)
    x3 = (1+x2+2*x6)/(4)
    x4 = (-2+x1+x5)/(4)
    x5 = (1+x2+x4+x6)/(4)
    x6 = (2+x3+x5)/(4)
    x = np.array([x1, x2, x3, x4, x5, x6])

    # check if it is smaller than threshold
    dx = np.sqrt(np.dot(x-x_old, x-x_old))

    print("%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"%(k, x1, x2, x3, x4, x5, x6))
    if dx < epsilon:
        converged = True
        print('Converged!')
        break
    x_old = x
if not converged:
    print('Not converge, increase the # of iterations')
        
print(f"Number of iterations for Gauss-Seidel = {k}")  
print(f"approximate x = {x}")      
