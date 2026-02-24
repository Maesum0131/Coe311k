#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import time

def jacobi(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        
        x = x_new
        print(x)
    
    return x, max_iterations


A = np.array([
    [10, -1,  2,  0,  0],
    [-1, 11, -1,  3,  0],
    [ 2, -1, 10, -1,  0],
    [ 0,  3, -1,  8, -2],
    [ 0,  0,  0, -2,  9],
], dtype=float)

b = np.array([14, 30, 26, 25, 37], dtype=float)

x0 = np.zeros_like(b)
tol = 1e-6
max_iterations = 100

start = time.perf_counter()

solution, iterations = jacobi(A, b, x0, tol, max_iterations)

end = time.perf_counter()
elapsed_time = end - start

print(f"solution: {solution}")
print(f"iterations: {iterations}")
print(f"time: {elapsed_time:.6e} seconds")


# In[9]:


import numpy as np
import time

def gauss_seidel(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    
    for k in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k

        x = x_new
        print(x)

    return x, max_iterations


A = np.array([
    [10, -1,  2,  0,  0],
    [-1, 11, -1,  3,  0],
    [ 2, -1, 10, -1,  0],
    [ 0,  3, -1,  8, -2],
    [ 0,  0,  0, -2,  9],
], dtype=float)

b = np.array([14, 30, 26, 25, 37], dtype=float)

x0 = np.zeros_like(b)
tol = 1e-6
max_iterations = 100

start = time.perf_counter()

solution, iterations = gauss_seidel(A, b, x0, tol, max_iterations)

end = time.perf_counter()
elapsed_time = end - start

print(f"solution: {solution}")
print(f"iterations: {iterations}")
print(f"time: {elapsed_time:.6e} seconds")


# In[10]:


import numpy as np
import time

def jacobi(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()

    for k in range(max_iterations):
        x_new = np.zeros_like(x)

        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k

        x = x_new
        print(x)

    return x, max_iterations


A = np.array([
    [1, 2, 3, 0, 0],
    [2, 1, 2, 3, 0],
    [3, 2, 1, 2, 3],
    [0, 3, 2, 1, 2],
    [0, 0, 3, 2, 1]
], dtype=float)

b = np.array([14, 22, 33, 26, 22], dtype=float)

x0 = np.zeros_like(b)
tol = 1e-6
max_iterations = 100

start = time.perf_counter()

solution, iterations = jacobi(A, b, x0, tol, max_iterations)

end = time.perf_counter()
elapsed_time = end - start

print(f"solution: {solution}")
print(f"iterations: {iterations}")
print(f"time: {elapsed_time:.6e} seconds")


# In[11]:


import numpy as np
import time

def gauss_seidel(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()

    for k in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k

        x = x_new
        print(x)

    return x, max_iterations


A = np.array([
    [1, 2, 3, 0, 0],
    [2, 1, 2, 3, 0],
    [3, 2, 1, 2, 3],
    [0, 3, 2, 1, 2],
    [0, 0, 3, 2, 1]
], dtype=float)

b = np.array([14, 22, 33, 26, 22], dtype=float)

x0 = np.zeros_like(b)
tol = 1e-6
max_iterations = 100

start = time.perf_counter()

solution, iterations = gauss_seidel(A, b, x0, tol, max_iterations)

end = time.perf_counter()
elapsed_time = end - start

print(f"solution: {solution}")
print(f"iterations: {iterations}")
print(f"time: {elapsed_time:.6e} seconds")


# In[ ]:




