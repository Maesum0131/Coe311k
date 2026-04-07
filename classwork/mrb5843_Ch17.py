#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Define parameters
h = 0.1  # step size
x_start, x_end = 0, 10
num_steps = int((x_end - x_start) / h)

x_values = np.linspace(x_start, x_end, num_steps + 1)
y_values = np.zeros(num_steps + 1)

# Initial condition
y_values[0] = 1  # y(0) = 1

# Function representing dy/dx = -2y
def f(x, y):
    return -2 * y

# Implement Euler's method
for i in range(num_steps):
    y_values[i + 1] = y_values[i] + h * f(x_values[i], y_values[i])


# Exact solution function
def exact_solution(x):
    # dy/dx = -2y, y(0) = 1  =>  y(x) = e^(-2x)
    return np.exp(-2 * x)


# Function to run Euler + compute local and global error
def compute_errors(h_val):
    n = int((x_end - x_start) / h_val)
    x_vals = np.linspace(x_start, x_end, n + 1)
    y_euler = np.zeros(n + 1)
    y_euler[0] = 1
    for i in range(n):
        y_euler[i + 1] = y_euler[i] + h_val * f(x_vals[i], y_euler[i])

    y_exact = exact_solution(x_vals)
    local_error = np.abs(y_euler - y_exact)     # error at each step
    global_error = np.mean(local_error)         # average error across interval
    return x_vals, y_euler, y_exact, local_error, global_error


# Plotting 
step_sizes = [0.1, 0.5, 1.0, 1.5]
colors = ['steelblue', 'darkorange', 'green', 'red']

fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle("Euler's Method Challenge: dy/dx = -2y", fontsize=16, fontweight='bold')

# Euler approximations vs exact solution
ax1 = axes[0]
x_exact_dense = np.linspace(x_start, x_end, 1000)
ax1.plot(x_exact_dense, exact_solution(x_exact_dense), 'k-', lw=2.5,
         label='Exact: $e^{-2x}$', zorder=5)

for h_val, color in zip(step_sizes, colors):
    x_v, y_e, _, _, _ = compute_errors(h_val)
    ax1.plot(x_v, y_e, '--o', color=color, markersize=3,
             label=f'Euler h={h_val}', alpha=0.85)

ax1.set_title("Euler Approximation vs Exact Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim(-2, 2)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', lw=0.8)

# Global (average) error vs h
ax2 = axes[1]
global_errors = []
for h_val in step_sizes:
    _, _, _, _, g_err = compute_errors(h_val)
    global_errors.append(g_err)

ax2.plot(step_sizes, global_errors, 'bo-', markersize=8, lw=2)
for h_val, err in zip(step_sizes, global_errors):
    ax2.annotate(f'{err:.4f}', (h_val, err), textcoords="offset points",
                 xytext=(5, 8), fontsize=9)

ax2.set_title("Global (Average) Error vs Step Size h")
ax2.set_xlabel("Step Size h")
ax2.set_ylabel("Mean |Euler - Exact|")
ax2.grid(True, alpha=0.3)

# Local error at each x
ax3 = axes[2]
for h_val, color in zip(step_sizes, colors):
    x_v, _, _, l_err, _ = compute_errors(h_val)
    ax3.plot(x_v, l_err, '-', color=color, label=f'h={h_val}', alpha=0.85)

ax3.set_title("Local Error at Each x  |Euler - Exact|")
ax3.set_xlabel("x")
ax3.set_ylabel("Local Error")
ax3.set_ylim(bottom=0)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("euler_challenge.png", dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:




