#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


k = 1.0       
h = 0.01       
x_start = 0    
x_end = 10     
y_0 = 1.0      
v_0 = 0.0      

# Challenge part 1

h_values = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
colors = cm.plasma(np.linspace(0, 0.9, len(h_values)))

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Exact analytical solution for reference
x_ref = np.linspace(x_start, x_end, 1000)
axes[0].plot(x_ref, np.cos(np.sqrt(k) * x_ref), 'k-', linewidth=2.5,
             label='Exact solution: cos(x)', zorder=10)

for h_val, color in zip(h_values, colors):

    # Create arrays to store x, y, and v values 
    
    x_values = np.arange(x_start, x_end + h_val, h_val)
    y_values = np.zeros(len(x_values))
    v_values = np.zeros(len(x_values))

    # Set initial conditions
    
    y_values[0] = y_0
    v_values[0] = v_0

    # Euler's Method loop 
    
    for i in range(1, len(x_values)):
        # Current values
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]
        v_n = v_values[i - 1]
        
        
        y_values[i] = y_n + h_val * v_n
        v_values[i] = v_n + h_val * (-k * y_n)

    axes[0].plot(x_values, y_values, label=f'h={h_val}', color=color,
                 alpha=0.85, linewidth=1.4)

axes[0].set_title('Challenge Part 1: All h values on same canvas (k=1.0)',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y(x)')
axes[0].legend(loc='upper right', ncol=3, fontsize=8)
axes[0].grid(True)
axes[0].set_ylim(-5, 5)

# Challenge part 2
k = 10.0       # Changed as per challenge
h = 0.01       # Professor's original step size

# Create arrays to store x, y, and v values 
x_values = np.arange(x_start, x_end + h, h)
y_values = np.zeros(len(x_values))
v_values = np.zeros(len(x_values))

# Set initial conditions 

y_values[0] = y_0
v_values[0] = v_0

# Euler's Method loop 

for i in range(1, len(x_values)):
    # Current values
    x_n = x_values[i - 1]
    y_n = y_values[i - 1]
    v_n = v_values[i - 1]
    # Update equations
    y_values[i] = y_n + h * v_n
    v_values[i] = v_n + h * (-k * y_n)

# Exact solution for k=10

x_ref2 = np.linspace(x_start, x_end, 5000)
axes[1].plot(x_ref2, np.cos(np.sqrt(k) * x_ref2), 'k-', linewidth=2.5,
             label='Exact solution: cos(√10·x)', zorder=10)

# Plot the results 

axes[1].plot(x_values, y_values, color='crimson', linewidth=1.5,
             label=f'Position y(x)  [k={k}, h={h}]')
axes[1].set_title('Challenge Part 2: k=10, h=0.01 — Euler amplitude drifts vs exact',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y and v')
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.tight_layout(pad=3)
plt.show()

