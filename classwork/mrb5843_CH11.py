#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define known data points

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 0, 1, 0, 1])

# Cubic Spline

cs = CubicSpline(x, y, bc_type='natural')

# Generate smooth x values

x_interp = np.linspace(0, 5, 100)
y_spline = cs(x_interp)

# Evaluate spline at original points for metrics

y_spline_points = cs(x)

# Polynomial Fit

degree = 5
coefficients = np.polyfit(x, y, degree)
poly = np.poly1d(coefficients)

y_poly = poly(x_interp)
y_poly_points = poly(x)

# metrics

# CubicSpline

mse_spline = np.mean((y - y_spline_points)**2)

ss_res_spline = np.sum((y - y_spline_points)**2)

ss_total = np.sum((y - np.mean(y))**2)

r2_spline = 1 - (ss_res_spline / ss_total)

# Polyfit

mse_poly = np.mean((y - y_poly_points)**2)

ss_res_poly = np.sum((y - y_poly_points)**2)

r2_poly = 1 - (ss_res_poly / ss_total)


print("CubicSpline Results")
print("MSE:", mse_spline)
print("R-squared:", r2_spline)
print()

print("Polyfit Results")
print("MSE:", mse_poly)
print("R-squared:", r2_poly)
print()

# Determine better model
if mse_spline < mse_poly:
    print("CubicSpline fits better")
else:
    print("Polyfit fits better")

# Plot Comparison 

plt.figure(figsize=(8,5))

plt.scatter(x, y, color='black', label="Data Points")

plt.plot(x_interp, y_spline, color='blue', label="CubicSpline")

plt.plot(x_interp, y_poly, color='red', linestyle='--', label="Polyfit")


plt.title("Polyfit vs CubicSpline Comparison")

plt.xlabel("x")

plt.ylabel("y")

plt.legend()

plt.grid(True)

plt.show()


# In[ ]:




