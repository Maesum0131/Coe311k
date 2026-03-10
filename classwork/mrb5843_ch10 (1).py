#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Step 1 from least squares problem
np.random.seed(0)
x = np.linspace(-5, 5, 50)
y = 2 * x**2 + 3 * x + np.random.randn(50) * 10

# Linear fit 1 degree 
degree = 1
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)
y_pred_linear = polynomial(x)

mse_linear = np.mean((y - y_pred_linear)**2)

ss_total = np.sum((y - np.mean(y))**2)
ss_res_linear = np.sum((y - y_pred_linear)**2)
r_squared_linear = 1 - (ss_res_linear / ss_total)

print("Linear MSE:", mse_linear)
print("Linear R-squared:", r_squared_linear)
print()

# cubic fit third degree 
degree = 3
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)
y_pred_cubic = polynomial(x)

mse_cubic = np.mean((y - y_pred_cubic)**2)

ss_res_cubic = np.sum((y - y_pred_cubic)**2)
r_squared_cubic = 1 - (ss_res_cubic / ss_total)

print("Cubic MSE:", mse_cubic)
print("Cubic R-squared:", r_squared_cubic)
print()

if mse_cubic < mse_linear:
    print("Cubic fits better")
else:
    print("Linear fits better")

# Plot
plt.scatter(x, y, label="Data points")
plt.plot(x, y_pred_linear, label="Linear fit")
plt.plot(x, y_pred_cubic, label="Cubic fit")
plt.legend()
plt.show()



# In[3]:


# crank up

# start at cubic
for degree in range(3, 11):

    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    y_pred = polynomial(x)

    mse = np.mean((y - y_pred)**2)

    ss_res = np.sum((y - y_pred)**2)
    ss_total = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_total)

    print("Degree:", degree)
    print("MSE:", mse)
    print("R-squared:", r_squared)
    print()

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.title(f"Degree {degree} (Up)")
    plt.show()



# In[4]:



# challenge b

# crank back down
for degree in range(3, 0, -1):

    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    y_pred = polynomial(x)

    mse = np.mean((y - y_pred)**2)

    ss_res = np.sum((y - y_pred)**2)
    ss_total = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_total)

    print("Degree:", degree)
    print("MSE:", mse)
    print("R-squared:", r_squared)
    print()

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.title(f"Degree {degree} (Down)")
    plt.show()


# In[5]:


# part c
import numpy as np
import matplotlib.pyplot as plt

# Same data from Part A

np.random.seed(0)
x = np.linspace(-5, 5, 50)
y = 2 * x**2 + 3 * x + np.random.randn(50) * 10

# Choose polynomial degree

degree = 3

# Create Vandermonde matrix

V = np.vander(x, degree + 1)

# Solve for coefficients using Least Squares

coefficients = np.linalg.lstsq(V, y, rcond=None)[0]

# Predict y values
y_pred = V @ coefficients

# Compute MSE
mse = np.mean((y - y_pred)**2)

# Compute R-squared
ss_res = np.sum((y - y_pred)**2)
ss_total = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_total)

# Print results

print("Vandermonde Polynomial Fit (Degree 3)")

print("Coefficients:", coefficients)

print("MSE:", mse)

print("R-squared:", r_squared)

# Plot results

plt.scatter(x, y, label="Data Points")
plt.plot(x, y_pred, color="red", label="Vandermonde Fit (Degree 3)")
plt.title("Polynomial Fit Using Vandermonde Matrix")
plt.legend()
plt.show()


# In[ ]:




