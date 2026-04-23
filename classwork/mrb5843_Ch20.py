#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define f
def f(t, y):
    return y**2 - 10

# Derivative of f with respect to y (needed for Newton-Raphson)
def fprime(y):
    return 2 * y

# Newton-Raphson for single step with Backward Euler
def backward_euler_step(f, y, t, h, tol=1e-6, max_iter=50):
    y_next = y  # Initial guess
    for _ in range(max_iter):
        g_val = y_next - y - h * f(t + h, y_next)
        g_prime_val = 1 - h * fprime(y_next)
        if abs(g_prime_val) < tol:  # Avoid division by zero
            break
        y_new = y_next - g_val / g_prime_val
        if abs(y_new - y_next) < tol:
            return y_new  # Converged
        y_next = y_new
    return y_next

# Adaptive Backward Euler
def backward_euler_adaptive(f, y0, t0, tf, h_init, tol=1e-6, max_iter=50):
    t_values = [t0]
    y_values = [y0]
    h = h_init
    t = t0
    y = y0

    while t < tf:
        if t + h > tf:  # Adjust step size to not overshoot
            h = tf - t

        # Single step with h
        y_full = backward_euler_step(f, y, t, h, tol, max_iter)

        # Two half-steps with h/2
        h_half = h / 2
        y_half_1 = backward_euler_step(f, y, t, h_half, tol, max_iter)
        y_half_2 = backward_euler_step(f, y_half_1, t + h_half, h_half, tol, max_iter)

        # Error estimation
        error = abs(y_full - y_half_2)

        # Adjust step size
        if error > tol:
            h /= 2  # Decrease step size
            continue  # Retry the step with smaller h
        elif error < tol / 2:
            h *= 2  # Increase step size for efficiency

        # Accept the step
        t += h
        y = y_half_2
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


y0 = 4.0   # Initial condition
t0 = 0.0
tf = 2.0
h_init = 0.1

t_vals, y_vals = backward_euler_adaptive(f, y0, t0, tf, h_init)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_vals, y_vals, 'b-o', markersize=4, label='Adaptive Backward Euler')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Adaptive Backward Euler: y' = y² - 10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("backward_euler_adaptive.png", dpi=150)
plt.show()

print(f"Steps taken: {len(t_vals) - 1}")
print(f"Final value: y({tf}) = {y_vals[-1]:.6f}")


# In[ ]:




