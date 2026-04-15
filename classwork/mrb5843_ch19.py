#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
 
# Parameters
k = 1.0
h_values = [0.01, 0.1, 0.5, 1.0]
t_start = 0
t_end = 20
y_0 = 1.0
v_0 = 0.0
 
def check_stability(h, k):
    """Construct coefficient matrix A and compute eigenvalues."""
    A = np.array([[1,      h   ],
                  [-h * k, 1   ]])
    eigenvalues = np.linalg.eigvals(A)
    is_stable = all(np.abs(eig) <= 1 for eig in eigenvalues)
    return eigenvalues, is_stable
 
def euler_method(h, k, t_start, t_end, y_0, v_0):
    """Simulate Euler's Method for a 2nd-order ODE (simple harmonic oscillator)."""
    t_values = np.arange(t_start, t_end + h, h)
    y_values = np.zeros(len(t_values))
    v_values = np.zeros(len(t_values))
 
    y_values[0] = y_0
    v_values[0] = v_0
 
    for i in range(1, len(t_values)):
        y_n = y_values[i - 1]
        v_n = v_values[i - 1]
        y_values[i] = y_n + h * v_n
        v_values[i] = v_n + h * (-k * y_n)
 
    return t_values, y_values
 
# Output & Plot 
plt.figure(figsize=(12, 6))
 
for h in h_values:
    eigenvalues, is_stable = check_stability(h, k)
    magnitudes = np.abs(eigenvalues)
 
    # Print results to console
    print(f"Step size h = {h}")
    print(f"  Eigenvalues : {eigenvalues}")
    print(f"  |λ|         : {magnitudes}")
    if is_stable:
        print("  → Solution is Stable")
    else:
        print("  → Solution may not be Entirely Stable ;)")
    print()
 
    # Simulate and plot
    t, y = euler_method(h, k, t_start, t_end, y_0, v_0)
    label = f"h={h} ({'Stable' if is_stable else 'Unstable'})"
    plt.plot(t, y, label=label)
 
plt.xlabel("Time (t)")
plt.ylabel("Displacement (y)")
plt.title("Stability Analysis of Euler's Method for a 2nd-Order ODE\n(Simple Harmonic Oscillator: d²y/dt² = -ky)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




