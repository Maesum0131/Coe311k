#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# Data points
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1.2, 2.8, 3.6, 4.5, 5.1])

# Stack the design matrix with a column of ones (for the intercept)
A = np.vstack([X, np.ones(len(X))]).T

print("X:")
print(X)

print("Y:")
print(Y)

print("A:")
print(A)


# In[3]:


# Solve using least squares
m, c = np.linalg.lstsq(A, Y, rcond=None)[0]

print(f"Slope: {m}, Intercept: {c}")


# In[4]:


import matplotlib.pyplot as plt

Y_line = m * X + c

plt.scatter(X, Y, label="Data points")
plt.plot(X, Y_line, label="Best fit line")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()


# In[5]:


# Randomize the vectors
perm = np.random.permutation(len(X))

X_shuffled = X[perm]
Y_shuffled = Y[perm]

print(X_shuffled)
print(Y_shuffled)


# In[6]:


A_shuffled = np.vstack([X_shuffled, np.ones(len(X_shuffled))]).T
m2, c2 = np.linalg.lstsq(A_shuffled, Y_shuffled, rcond=None)[0]

print(f"New Slope: {m2}, New Intercept: {c2}")


# In[ ]:




