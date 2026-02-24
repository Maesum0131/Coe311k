#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


def taylor_sin(x, n_terms):
 
    
    # Ensure x is treated as a NumPy array 
    x = np.asarray(x, dtype=float)
    
    # array of zeros with the same shape as x
    s = np.zeros_like(x)
    
    # Taylor series sum

    for n in range(n_terms):
        s += ((-1)**n) * x**(2*n + 1) / math.factorial(2*n + 1)
    
    return s



# Main program


# step size
x = np.arange(0, 10.0 + 0.1, 0.1)

# Compute the exact sin(x) using NumPy
sin_exact = np.sin(x)

# store approximations and errors
approxs = {}
errors = {}

# Compute approximations and absolute errors
for N in [1, 2, 3, 4]:
    # Taylor approximation with N terms
    approxs[N] = taylor_sin(x, N)
    
    # Absolute error compared to exact sin(x)
    errors[N] = np.abs(approxs[N] - sin_exact)


# Plot

plt.figure()

# Plot error curves for each number of terms
for N in [1, 2, 3, 4]:
    plt.plot(x, errors[N], label=f"{N} term(s)")

plt.title("Truncation error for Taylor approximation of sin(x)")
plt.xlabel("x")
plt.ylabel("Absolute error")

# Use logarithmic scale to better visualize error decay
plt.yscale("log")

plt.legend()
plt.grid(True)
plt.show()


# The truncation error decreases as more terms are added to the Taylor series. Using only one term produces large errors for medium and large values of x. Adding additional terms significantly reduces the error.


# In[ ]:





# In[ ]:





# In[ ]:




