#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Function dy/dx = -2y
def f(x, y):
    return -2 * y

# Domain
x_start, x_end = 0, 2

# Step sizes from 0.01 to 1 
h_values = np.arange(0.01, 1.01, 0.05)

# Store baseline
baseline_x = None
baseline_y = None

plt.figure(figsize=(10, 6))

for idx, h in enumerate(h_values):

    num_steps = int((x_end - x_start) / h)
    x_values = np.linspace(x_start, x_end, num_steps + 1)
    y_values = np.zeros(num_steps + 1)

    # Initial condition
    y_values[0] = 1

    # Euler method
    for i in range(num_steps):
        y_values[i + 1] = y_values[i] + h * f(x_values[i], y_values[i])

    # Plot each run
    plt.plot(x_values, y_values, label=f"h={h:.2f}", alpha=0.6)

    # Save first run as baseline
    if idx == 0:
        baseline_x = x_values
        baseline_y = y_values
    else:
        # Interpolate baseline to match current grid
        baseline_interp = np.interp(x_values, baseline_x, baseline_y)

        # Compute pointwise error
        error = np.abs(y_values - baseline_interp)

        # Total accumulated error
        
        total_error = np.sum(error)

        print(f"\nStep size h = {h:.2f}")
        print("Pointwise error:", error)
        print("Total accumulated error:", total_error)

# Labels
plt.xlabel("x")
plt.ylabel("y")
plt.title("Euler Method with Varying Step Sizes")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:




