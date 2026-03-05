#!/usr/bin/env python
# coding: utf-8

# In[1]:


## own least squares method


# In[12]:


import numpy as np
import matplotlib.pyplot as plt


# Create X values
#sort x values
X = np.arange(0, 1.01, 0.01)
X = np.sort(X)

# Random slope and intercept
m_true = np.random.uniform(-5,5)
b_true = np.random.uniform(-5,5)

# Random between 0 and 1
noise = np.random.uniform(0,1,len(X))

# Generate Y values
Y = m_true * X + b_true + noise

print("Original Equation:")
print(f"y = {m_true:.4f}x + {b_true:.4f}")

# Design Matrix 

A = np.vstack([X, np.ones(len(X))]).T

# compare NumPy Least Squares

m_numpy, b_numpy = np.linalg.lstsq(A, Y, rcond=None)[0]

print("\nNumPy Least Squares:")
print(f"Slope = {m_numpy:.6f}")  
print(f"Intercept = {b_numpy:.6f}")

# my Least Squares

n = len(X)

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X*Y)
sum_x2 = np.sum(X**2)

m_custom = (n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2)
b_custom = (sum_y - m_custom*sum_x)/n

print("\nCustom Least Squares:")
print(f"Slope = {m_custom:.6f}")
print(f"Intercept = {b_custom:.6f}")
print(f"Slope difference: {abs(m_custom - m_numpy):.12e}")
print(f"Intercept difference: {abs(b_custom - b_numpy):.12e}")

# Predictions

Y_pred = m_custom*X + b_custom

print("\nPredictions:")
print(Y_pred)

# Residual errors

residuals = Y - Y_pred

print("\nResiduals list:")
print(residuals)

# Largest residual
max_index = np.argmax(np.abs(residuals))

# Smallest residual
min_index = np.argmin(np.abs(residuals))

print("\nLargest Residual:")
print("X =", X[max_index])
print("Residual =", residuals[max_index])

print("\nSmallest Residual:")
print("X =", X[min_index])
print("Residual =", residuals[min_index])


# In[13]:


# Plot Results

plt.scatter(X,Y,label="Data Points")

plt.plot(X,Y_pred,label="Custom Least Squares",linewidth=1.5)

plt.plot(X,m_numpy*X + b_numpy,
         linestyle="--",
         label="NumPy Least Squares")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Least Squares Comparison")

plt.legend()
plt.grid(True)

plt.show()


# In[ ]:




