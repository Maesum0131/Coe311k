#!/usr/bin/env python
# coding: utf-8

# In[10]:



#7a
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
    return x,max_iterations

   


# In[7]:



   A = np.array([
   [0.582745, 0.48,     0.10,    0,       0      ],
   [0.48,     1.044129, 0.46,    0.10,    0      ],
   [0.10,     0.46,     1.10431, 0.44,    0.10   ],
   [0,        0.10,     0.44,    0.963889,0.42   ],
   [0,        0,        0.10,    0.42,    0.522565]
], dtype=float)

b = np.array([1.162745, 2.084129, 2.20431, 1.923889, 1.042565], dtype=float)

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


# In[13]:


A = np.array([
    [0.582745, 0.48,     0.10,    0,       0      ],
    [0.48,     1.044129, 0.46,    0.10,    0      ],
    [0.10,     0.46,     1.10431, 0.44,    0.10   ],
    [0,        0.10,     0.44,    0.963889,0.42   ],
    [0,        0,        0.10,    0.42,    0.522565]
], dtype=float)

b = np.array([1.162745, 2.084129, 2.20431, 1.923889, 1.042565], dtype=float)

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


# In[14]:



#7b
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
    return x, max_iterations


# In[15]:


A = np.array([[1, -1], [1, 1.0001]])
b = np.array([1, 1])

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


# In[16]:


import numpy as np

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


# In[17]:


A = np.array([[1, -1], [1, 1.0001]])
b = np.array([1, 1])
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


# In[18]:


#7c
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
    return x, max_iterations


# In[19]:


A = np.array([[1, -1], [1, 1.0001]])
b = np.array([1, 1.0001])

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


# In[24]:


import numpy as np

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


# In[25]:


A = np.array([[1, -1], [1, 1.0001]])
b = np.array([1, 1.0001])
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


# In[26]:


#7d
import numpy as np

A = np.array([
    [0.582745, 0.48,     0.10,    0,       0      ],
    [0.48,     1.044129, 0.46,    0.10,    0      ],
    [0.10,     0.46,     1.10431, 0.44,    0.10   ],
    [0,        0.10,     0.44,    0.963889,0.42   ],
    [0,        0,        0.10,    0.42,    0.522565]
], dtype=float)

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]

    for i in range(n):
        diagonal = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)

        if diagonal <= row_sum:
            return False

    return True

print("Is the matrix diagonally dominant?", is_diagonally_dominant(A))


# In[ ]:




