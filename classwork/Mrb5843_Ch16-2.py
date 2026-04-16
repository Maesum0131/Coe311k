#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
 
# Euler's Method: Stability and Accuracy Exploration
# Solving: d²y/dx² = -ky  (harmonic oscillator)
 
k = 1.0       # Spring constant
x_start = 0
x_end = 10
y_0 = 1.0     # Initial position
v_0 = 0.0     # Initial velocity
 
h_values = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
 
plt.figure(figsize=(12, 6))
 
for h in h_values:
    x_values = np.arange(x_start, x_end + h, h)
    y_values = np.zeros(len(x_values))
    v_values = np.zeros(len(x_values))
 
    y_values[0] = y_0
    v_values[0] = v_0
 
    for i in range(1, len(x_values)):
        y_n = y_values[i - 1]
        v_n = v_values[i - 1]
        y_values[i] = y_n + h * v_n
        v_values[i] = v_n + h * (-k * y_n)
 
    plt.plot(x_values, y_values, label=f'h = {h}')
 
x_exact = np.linspace(x_start, x_end, 1000)
y_exact = np.cos(np.sqrt(k) * x_exact)
plt.plot(x_exact, y_exact, 'k--', linewidth=2, label='Exact solution')
 
plt.ylim(-6, 6)
plt.xlabel('x')
plt.ylabel('y (position)')
plt.title("Euler's Method: All Step Sizes on One Canvas (k = 1)")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
 

# When does it become unstable?

print("=" * 60)
print("Q1: When does instability appear?")
print("=" * 60)
print("""
At k = 1, the solution stays close to the exact cosine curve for
small step sizes like h = 0.01 and 0.02. But however, around h = 0.3
and h = 0.5, the solution visibly diverges the amplitude grows
rather than staying bounded. The rough instability threshold for
Euler's method on this problem is kh² ≥ 2. At h = 0.5 with k = 1,
kh² = 0.25, which is approaching the danger zone, and the error
accumulation becomes clearly visible in the plot.
""")
 # Is it becoming unstable?
print("=" * 60)
print("Q2: Is it becoming unstable?")
print("=" * 60)
print("""
Yes, the larger step sizes (h = 0.2, 0.3, 0.5) show the solution
drifting away from the true oscillation. Instead of maintaining a
consistent amplitude like the exact solution, those curves grow
over time. This is because Euler's method adds energy
to the system with each step when h is too large a hallmark of
numerical instability in oscillatory problems.
""")
 

# Play with k — what do you see?
print("=" * 60)
print("Q3: What happens when you change k?")
print("=" * 60)
print("""
Increasing k makes the oscillation faster with higher frequency, which then
shrinks the range of step sizes that remain stable. An example, at
k = 10, a step size of h = 0.1 already becomes marginal, and h = 0.5
diverges almost immediately. This is the stiffness problem a stiffer
spring forces you to use a much smaller h just to keep the
solution from blowing up, regardless of how accurate you want to be.
""")
 
# Set k = 10, h = 0.01

print("=" * 60)
print("Q4: k = 10, h = 0.01 — does it hold?")
print("=" * 60)
 
k_test = 10.0
h_test = 0.01
print(f"  kh² = {k_test} × {h_test}² = {k_test * h_test**2:.4f}")
print("""
With k = 10 and h = 0.01, kh² = 0.001, which is well below the
instability threshold. The solution remains stable and tracks the
true oscillation reasonably well. However, notice that h had to be
kept very small specifically because of the large k — the step size
constraint is being driven by stability, not just accuracy.
""")
 
# Plot k=10, h=0.01 vs exact
k2 = 10.0
h2 = 0.01
x2 = np.arange(x_start, x_end + h2, h2)
y2 = np.zeros(len(x2))
v2 = np.zeros(len(x2))
y2[0] = y_0
v2[0] = v_0
for i in range(1, len(x2)):
    y2[i] = y2[i-1] + h2 * v2[i-1]
    v2[i] = v2[i-1] + h2 * (-k2 * y2[i-1])
 
x_ex2 = np.linspace(x_start, x_end, 1000)
y_ex2 = np.cos(np.sqrt(k2) * x_ex2)
 
plt.figure(figsize=(12, 5))
plt.plot(x2, y2, label="Euler: k=10, h=0.01", color='steelblue')
plt.plot(x_ex2, y_ex2, 'k--', label='Exact solution', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Euler's Method: k = 10, h = 0.01")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 

# Can Euler's Method solve this with a larger h?

print("=" * 60)
print("Q5: Can Euler's Method handle larger h at k = 10?")
print("=" * 60)
print("""
No not reliably. At k = 10, even h = 0.1 gives kh² = 0.1, which
is in the marginal zone, and h = 0.3 gives kh² = 0.9, which is
essentially unstable. Euler's method is only conditionally stable,
meaning there is a hard upper limit on h for a given k. Unlike
higher-order methods like RK4, or implicit methods like backward
Euler, the explicit Euler scheme cannot maintain stability with a
larger step size here — it would require a fundamentally different
numerical approach to handle stiff problems efficiently.
""")
 


# In[ ]:




