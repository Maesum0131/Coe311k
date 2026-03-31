#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


f = lambda x: np.sin(x)
a, b = 0, 2 * np.pi
N = 50

# uniform grid
x = np.linspace(a, b, N)
y = f(x)

# Step size
h = (b - a) / (N - 1)

# Central diff
df_central = (f(x + h) - f(x - h)) / (2 * h)

# Forward diff
df_forward = (f(x + h) - f(x)) / h

# Backward Diff
df_backward = (f(x) - f(x - h)) / h

# derivative
df_exact = np.cos(x)

plt.figure(figsize=(10, 8))

#  Exact
plt.subplot(2, 2, 1)
plt.plot(x, df_exact)
plt.title("Exact Derivative (cos(x))")

#  Forward
plt.subplot(2, 2, 2)
plt.plot(x, df_forward)
plt.title("Forward Difference")

#  Backward
plt.subplot(2, 2, 3)
plt.plot(x, df_backward)
plt.title("Backward Difference")

#  Central
plt.subplot(2, 2, 4)
plt.plot(x, df_central)
plt.title("Central Difference")

plt.tight_layout()
plt.show()


# In[ ]:




