from numpy import array
from numpy import linalg as LA
from numpy import inf
import copy

# given linear system

A = array([[4.0, -1, 0, -2, 0, 0], [-1, 4, -1, 0, -2, 0], [0, -1, 4, 0, 0, -2], [-1, 0, 0, 4, -1, 0], [0, -1, 0, -1, 4, -1], [0, 0, -1, 0, -1, 4]])
b = array([-1.0,0,1,-2,1,2])

m = len(b) # return the array is the same length as b
x = array([0.0,0,0,0,0,0]) #initial vector
tol = 5*10**-6 # iteration max limit

n = 0 # iteration counter
xn = 1 # initial tolerance

while xn > tol:
    x0 = copy.copy(x)
    x[0] = (b[0] - ( A[0,1]*x[1]) )/A[0,0]

    for i in range(1,m-1):
        x[i] = ( b[i]- ( A[i, i-1]*x[i-1] + A[i,i+1]*x[i+1] ) )/A[i,i]
        
    x[m-1] = (b[m-1] - ( A[m-1,m-2]*x[m-2] ) )/A[m-1,m-1]
   
    #xn = LA.norm(x-x0,1) 
    #xn = LA.norm(x-x0,2) 
    xn = LA.norm(x-x0,inf)
    n = n + 1

print(f" Number of iterations for Gauss-Seidel = {n}")  
print(f"approximate x = {x}")      
