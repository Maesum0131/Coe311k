#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Pivoting
        max_row = max(range(i, n), key=lambda x: abs(A[x][i]))
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Make the rows below this one 0 in current column
        for j in range(i + 1, n):
            ratio = A[j][i] / A[i][i]
            A[j] = [A[j][k] - ratio * A[i][k] for k in range(n)]
            b[j] -= ratio * b[i]
    
    # Backward substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


# In[2]:


import numpy as np
from scipy.linalg import lu

# Example matrix
A = np.array([[2, 3, 1],
              [4, 7, -1],
              [-2, 1, 2]])
# Perform LU Decomposition
P, L, U = lu(A)

print("Matrix A:\n", A)
print("Permutation Matrix P:\n", P)
print("Lower Triangular Matrix L:\n", L)
print("Upper Triangular Matrix U:\n", U)


# In[ ]:




