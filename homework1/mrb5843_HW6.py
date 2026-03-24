#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Build the system (A, B, C, D)
def build_system(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros(n-2)
    B = np.zeros(n-2)
    C = np.zeros(n-2)
    D = np.zeros(n-2)
    for i in range(1, n-1):
        A[i-1] = h[i-1]
        B[i-1] = 2 * (h[i-1] + h[i])
        C[i-1] = h[i]
        slope1 = (y[i] - y[i-1]) / h[i-1]
        slope2 = (y[i+1] - y[i]) / h[i]
        D[i-1] = 6 * (slope2 - slope1)
    return A, B, C, D


# In[11]:


# Step 2: Solve using Thomas Algorithm (fixed off-by-one on A[i-1])
def solve_tridiagonal(A, B, C, D):
    n = len(D)
    B = B.copy()
    D = D.copy()
    # Forward elimination
    for i in range(1, n):
        factor = A[i-1] / B[i-1]  # FIXED: was A[i]
        B[i] = B[i] - factor * C[i-1]
        D[i] = D[i] - factor * D[i-1]
    # Back substitution
    M = np.zeros(n)
    M[-1] = D[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i] * M[i+1]) / B[i]
    return 


# In[12]:


# Step 3: Put M together (natural spline: M[0] = M[-1] = 0)
def get_M(x, y):
    A, B, C, D = build_system(x, y)
    M_inner = solve_tridiagonal(A, B, C, D)
    M = np.zeros(len(x))
    M[1:-1] = M_inner
    return M


# In[13]:


# Step 4: Evaluate spline at one point
def spline_value(x, y, M, x_val):
    for i in range(len(x)-1):
        if x[i] <= x_val <= x[i+1]:
            h = x[i+1] - x[i]
            a = M[i]   * (x[i+1] - x_val)**3 / (6*h)
            b = M[i+1] * (x_val  - x[i])**3   / (6*h)
            c = (y[i]   / h - M[i]   * h / 6) * (x[i+1] - x_val)
            d = (y[i+1] / h - M[i+1] * h / 6) * (x_val  - x[i])
            return a + b + c + d


# In[14]:


# Step 5: Generate full curve
def make_spline_curve(x, y):
    M = get_M(x, y)
    xs = []
    ys = []
    val = x[0]
    while val <= x[-1]:
        xs.append(val)
        ys.append(spline_value(x, y, M, val))
        val += 0.01
    return xs, ys


# In[19]:


# Example data (replace with yours)
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([0, 1, 0, 1, 0], dtype=float)

xs, ys = make_spline_curve(x, y)


# In[20]:


# Plot
plt.plot(xs, ys, '-', color='blue', linewidth=2, label="Cubic Spline")  # '-' forces a solid line
plt.scatter(x, y, color="red", zorder=5, s=100, label="Data Points")
plt.title("Cubic Spline")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




