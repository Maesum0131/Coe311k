#!/usr/bin/env python
# coding: utf-8

# In[8]:




import numpy as np



def setup_tridiagonal_matrix(x, y):

    n = len(x)                     # number of data points
    
    h = np.diff(x)                 # step sizes between x values

    # create empty arrays for the tridiagonal system
    A = np.zeros(n - 2)            # subdiagonal
    
    B = np.zeros(n - 2)            # main diagonal
    
    C = np.zeros(n - 2)            # superdiagonal
    
    D = np.zeros(n - 2)            # right-hand side

    # loop through interior points (skip first and last)
    
    for i in range(1, n - 1):

        A[i - 1] = h[i - 1]        # left interval length

        B[i - 1] = 2 * (h[i - 1] + h[i])   # combines both neighboring intervals

        C[i - 1] = h[i]            # right interval length

        # calculate slopes on each side of point i
        
        slope_left = (y[i] - y[i - 1]) / h[i - 1]
        
        slope_right = (y[i + 1] - y[i]) / h[i]

        # RHS represents change in slope (curvature)
        
        D[i - 1] = 6 * (slope_right - slope_left)

    return A, B, C, D


# In[10]:



x = np.array([0, 1, 2, 3, 4, 5])

y = np.array([0, 1, 0, 1, 0, 1])


A, B, C, D = setup_tridiagonal_matrix(x, y)

# Print results
print("Subdiagonal (A):", A)
print("Main diagonal (B):", B)
print("Superdiagonal (C):", C)
print("Right-hand side (D):", D)


# In[ ]:




