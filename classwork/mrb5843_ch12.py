#!/usr/bin/env python
# coding: utf-8

# In[2]:




import numpy as np

def setup_tridiagonal_matrix(x, y):

    n = len(x)
    h = np.diff(x)   # step sizes

    # initialize vectors
    A = np.zeros(n-2)   # subdiagonal
    B = np.zeros(n-2)   # main diagonal
    C = np.zeros(n-2)   # superdiagonal
    D = np.zeros(n-2)   # right-hand side

    for i in range(1, n-1):
        A[i-1] = h[i-1]
        B[i-1] = 2*(h[i-1] + h[i])
        C[i-1] = h[i]
        D[i-1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    return A, B, C, D


# example data
x = np.array([0,1,2,3,4,5])
y = np.array([0,1,1,0,1,0])

# compute matrices
A, B, C, D = setup_tridiagonal_matrix(x, y)

print("Subdiagonal (A):", A)
print("Main diagonal (B):", B)
print("Superdiagonal (C):", C)
print("Right-hand side (D):", D)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 0, 1, 0, 1])

# --- Cubic Spline ---
cs = CubicSpline(x, y, bc_type='natural')
x_interp = np.linspace(0, 5, 100)
y_spline = cs(x_interp)

# --- Polynomial Fit (degree 5 to pass through all 6 points) ---
coeffs = np.polyfit(x, y, deg=5)
y_poly = np.polyval(coeffs, x_interp)

# --- Plot both on the same graph ---
plt.plot(x, y, 'bo', label="Data points", zorder=5)
plt.plot(x_interp, y_spline, 'r-', label="Cubic Spline")
plt.plot(x_interp, y_poly, 'g--', label="Polynomial Fit (deg 5)")
plt.title("Cubic Spline vs Polynomial Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




