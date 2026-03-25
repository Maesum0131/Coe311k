#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import time

# Naive matrix multiplication 
def naive_multiply(A, B):
    n = len(A)
    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


# Experiment setup
matrix_nums = []          # 1 through 8
naive_times = []
numpy_times = []

size = 3   # starting size

for i in range(8):

    # generate random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    matrix_nums.append(i + 1)

    # time naive method
    start = time.perf_counter()
    naive_multiply(A, B)
    end = time.perf_counter()
    naive_times.append(end - start)

    # time numpy method
    start = time.perf_counter()
    np.dot(A, B)
    end = time.perf_counter()
    numpy_times.append(end - start)

    # increase size (double each time: 3 → 6 → 12 → ...)
    size *= 2


# Plot (based on professor's plotting style)
plt.plot(matrix_nums, naive_times, label='Naive (O(n^3))')
plt.plot(matrix_nums, numpy_times, label='NumPy (Optimized)')

plt.title('Matrix Multiplication Runtime Comparison')
plt.xlabel('Matrix Number (1 → 8)')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.show


# In[ ]:




