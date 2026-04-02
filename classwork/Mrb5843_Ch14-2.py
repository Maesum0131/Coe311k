#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Function
f = lambda x: np.sin(x)

# Interval and points
a, b = 0, 2 * np.pi
N = 50

# Grid
x = np.linspace(a, b, N)
y = f(x)

# Step size
h = (b - a) / (N - 1)

# Initialize arrays
df_forward = np.zeros_like(x)
df_backward = np.zeros_like(x)
df_central = np.zeros_like(x)

# Forward Difference (exclude last point)
df_forward[:-1] = (y[1:] - y[:-1]) / h

# Backward Difference (exclude first point)
df_backward[1:] = (y[1:] - y[:-1]) / h

# Central Difference (exclude first & last points)
df_central[1:-1] = (y[2:] - y[:-2]) / (2 * h)

# Exact derivative
df_exact = np.cos(x)

# Plot
plt.figure(figsize=(10, 8))

# Exact
plt.subplot(2, 2, 1)
plt.plot(x, df_exact)
plt.title("Exact: cos(x)")

# Forward
plt.subplot(2, 2, 2)
plt.plot(x, df_exact, label='Exact')
plt.plot(x, df_forward, '--', label='Forward')
plt.legend()
plt.title("Forward Difference")

# Backward
plt.subplot(2, 2, 3)
plt.plot(x, df_exact, label='Exact')
plt.plot(x, df_backward, '--', label='Backward')
plt.legend()
plt.title("Backward Difference")

# Central
plt.subplot(2, 2, 4)
plt.plot(x, df_exact, label='Exact')
plt.plot(x, df_central, '--', label='Central')
plt.legend()
plt.title("Central Difference")

plt.tight_layout()
plt.show()


# In[ ]:




