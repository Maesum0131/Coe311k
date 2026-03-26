#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Define Known Data Points

x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = np.array([0, 1, 0, 1, 0, 1], dtype=float)
n = len(x)

#Calculate Step Sizes

h = np.diff(x)


# Set Up the Tridiagonal System


def setup_tridiagonal_matrix(x, y, h):
    n = len(x)
    A = np.zeros(n - 2)  # subdiagonal
    B = np.zeros(n - 2)  # main diagonal
    C = np.zeros(n - 2)  # superdiagonal
    D = np.zeros(n - 2)  # right-hand side

    for i in range(1, n - 1):
        A[i - 1] = h[i - 1]
        B[i - 1] = 2 * (h[i - 1] + h[i])
        C[i - 1] = h[i]
        D[i - 1] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    return A, B, C, D

# Solve Using the Thomas Algorithm

def thomas_algorithm(A, B, C, D):
    """
    Solve the tridiagonal system A*x[i-1] + B*x[i] + C*x[i+1] = D
    using the Thomas algorithm (O(n) Gaussian elimination for tridiagonals).
    """
    n = len(B)
    # Work on copies so we don't modify the originals
    b = B.copy()
    d = D.copy()

    # Forward elimination
    for i in range(1, n):
        w = A[i - 1] / b[i - 1]
        b[i] = b[i] - w * C[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x_sol = np.zeros(n)
    x_sol[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x_sol[i] = (d[i] - C[i] * x_sol[i + 1]) / b[i]

    return x_sol

# Build and solve the system
A, B, C, D = setup_tridiagonal_matrix(x, y, h)
M_internal = thomas_algorithm(A, B, C, D)

# Natural spline boundary conditions: M[0] = M[n-1] = 0
M = np.zeros(n)
M[1:-1] = M_internal  # fill internal second derivatives

print("Second derivatives M:", M)



#   a_i = y[i]
#   b_i = (y[i+1]-y[i])/h[i] - h[i]/6 * (M[i+1] + 2*M[i])
#   c_i = M[i] / 2
#   d_i = (M[i+1] - M[i]) / (6*h[i])
#   S_i(xeval) = a + b*(xeval-x[i]) + c*(xeval-x[i])^2 + d*(xeval-x[i])^3

def evaluate_spline(xeval, x, y, h, M):
    """
    Evaluate the cubic spline at a single point xeval.
    Finds the correct interval, computes coefficients, returns spline value.
    """
    n = len(x)

    # Find the interval index i such that x[i] <= xeval < x[i+1]
    i = np.searchsorted(x, xeval, side='right') - 1
    # Clamp to valid range
    i = int(np.clip(i, 0, n - 2))

    dx = xeval - x[i]

    # Spline coefficients from second derivatives M
    a = y[i]
    b = (y[i + 1] - y[i]) / h[i] - h[i] / 6 * (M[i + 1] + 2 * M[i])
    c = M[i] / 2
    d = (M[i + 1] - M[i]) / (6 * h[i])

    return a + b * dx + c * dx**2 + d * dx**3

# Generate smooth x values and evaluate spline at each

x_plot = np.linspace(x[0], x[-1], 300)
y_plot = np.array([evaluate_spline(xi, x, y, h, M) for xi in x_plot])

#  Plot the Results

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'bo', markersize=8, label="Data Points", zorder=5)
plt.plot(x_plot, y_plot, 'r-', linewidth=2, label="Custom Cubic Spline (from scratch)")
plt.title("Custom Cubic Spline Interpolation (From Scratch)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




