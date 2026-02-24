#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import time
import matplotlib.pyplot as plt


# Jacobi method
def jacobi(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    errors = []

    for k in range(max_iter):
        x_new = np.zeros(n)

        # compute each component using old x values
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]

            x_new[i] = (b[i] - s) / A[i][i]

        err = np.linalg.norm(x_new - x, np.inf)
        errors.append(err)

        if err < tol:
            return x_new, k + 1, errors

        x = x_new

    return x, max_iter, errors


# In[ ]:


# Gauss-Seidel method

# edited from challenges
def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    errors = []

    for k in range(max_iter):
        x_old = x.copy()

        # update values immediately as they are computed
        for i in range(n):
            s1 = 0
            for j in range(i):
                s1 += A[i][j] * x[j]

            s2 = 0
            for j in range(i+1, n):
                s2 += A[i][j] * x_old[j]

            x[i] = (b[i] - s1 - s2) / A[i][i]

        err = np.linalg.norm(x - x_old, np.inf)
        errors.append(err)

        if err < tol:
            return x, k + 1, errors

    return x, max_iter, errors


# In[7]:


# system from homework
A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 3]], dtype=float)

b = np.array([15, 10, 10], dtype=float)

x0 = np.array([0.0, 0.0, 0.0]) # from the homework
max_iter = 1000
tolerances = [1e-3, 1e-6, 1e-9] # from the homework


for tol in tolerances:
    print("\n----------------------------------")
    print("Tolerance =", tol)

    # Jacobi
    start = time.time()
    sol_j, it_j, err_j = jacobi(A, b, x0, tol, max_iter)
    time_j = time.time() - start

    # Gauss-Seidel
    start = time.time()
    sol_gs, it_gs, err_gs = gauss_seidel(A, b, x0, tol, max_iter) # same from the challenges 
    time_gs = time.time() - start

    print("\nJacobi:")
    print("Solution:", sol_j)
    print("Iterations:", it_j)
    print("Time:", time_j)

    print("\nGauss-Seidel:")
    print("Solution:", sol_gs)
    print("Iterations:", it_gs)
    print("Time:", time_gs)
    
    # plot convergence shows all three convergences 
    plt.figure()
    plt.semilogy(err_j)
    plt.semilogy(err_gs)
    plt.legend(["Jacobi", "Gauss-Seidel"])
    plt.xlabel("Iteration")
    plt.ylabel("Infinity Norm Error")
    plt.title("Convergence (tol = " + str(tol) + ")")
    plt.show()


# In[6]:





# In[ ]:




