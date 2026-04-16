#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Parameters
x_start = 0
x_end = 10 * np.pi
y0 = 1  # y(0) = 1

# ODE: dy/dx = -2y*cos(x)
def f(x, y):
    return -2 * y * np.cos(x)

# Exact solution: y(x) = e^(-2*sin(x))
def exact_solution(x):
    return np.exp(-2 * np.sin(x))

# Euler's method
def euler_method(h):
    num_steps = int((x_end - x_start) / h)
    x_vals = np.linspace(x_start, x_end, num_steps + 1)
    y_vals = np.zeros(num_steps + 1)
    y_vals[0] = y0
    for i in range(num_steps):
        y_vals[i + 1] = y_vals[i] + h * f(x_vals[i], y_vals[i])
    return x_vals, y_vals

# Step sizes
step_sizes = [0.01, 0.05, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]

# Run Euler for each h, compute averages and errors
avg_y_values = []
avg_errors = []
local_errors_at_15 = []

for h in step_sizes:
    x_vals, y_vals = euler_method(h)
    y_exact = exact_solution(x_vals)
    avg_y_values.append(np.mean(y_vals))
    avg_errors.append(np.mean(np.abs(y_vals - y_exact)))
    idx = np.argmin(np.abs(x_vals - 15))
    local_errors_at_15.append(np.abs(y_vals[idx] - exact_solution(x_vals[idx])))

# Plotting

fig, axes = plt.subplots(3, 1, figsize=(13, 16))
fig.suptitle("Euler's Method: dy/dx = -2y·cos(x), y(0)=1",
             fontsize=15, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 1, len(step_sizes)))

# Euler solution for each h vs exact
ax1 = axes[0]
x_exact_dense = np.linspace(x_start, x_end, 5000)
ax1.plot(x_exact_dense, exact_solution(x_exact_dense), 'k-', lw=2,
         label='Exact solution', zorder=5)
for h, color in zip(step_sizes, colors):
    x_vals, y_vals = euler_method(h)
    ax1.plot(x_vals, y_vals, '-', color=color, lw=1.2, label=f'h={h}', alpha=0.8)
ax1.axvline(x=15, color='gray', linestyle='--', lw=1, label='x≈15')
ax1.set_title("Euler's Method for Each Step Size h")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim(-1, 4)
ax1.legend(fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

# Average error vs h 
ax2 = axes[1]
ax2.plot(step_sizes, avg_errors, 'bo-', markersize=8, lw=2)
for h, err in zip(step_sizes, avg_errors):
    ax2.annotate(f'{err:.4f}', (h, err), textcoords="offset points",
                 xytext=(5, 6), fontsize=8)
ax2.set_title("Average Error vs Step Size h")
ax2.set_xlabel("Step Size h")
ax2.set_ylabel("Mean |Euler - Exact|")
ax2.grid(True, alpha=0.3)

# Local error at x ≈ 15 vs h 
ax3 = axes[2]
ax3.plot(step_sizes, local_errors_at_15, 'rs-', markersize=8, lw=2)
for h, err in zip(step_sizes, local_errors_at_15):
    ax3.annotate(f'{err:.4f}', (h, err), textcoords="offset points",
                 xytext=(5, 6), fontsize=8)
ax3.set_title("Local Error at x ≈ 15 vs Step Size h")
ax3.set_xlabel("Step Size h")
ax3.set_ylabel("Local Error |Euler - Exact| at x≈15")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("homework7_euler.png", dpi=150, bbox_inches='tight')
plt.show()

# Summary table 
print(f"\n{'h':>6} | {'Avg y':>10} | {'Avg Error':>12} | {'Local Error @ x≈15':>20}")
print("-" * 58)
for h, avg_y, avg_err, loc_err in zip(step_sizes, avg_y_values, avg_errors, local_errors_at_15):
    print(f"{h:>6} | {avg_y:>10.5f} | {avg_err:>12.6f} | {loc_err:>20.6f}")


print("""
================================================================================
STABILITY ANALYSIS: dy/dx = -2y*cos(x)
================================================================================

What is our λ?
--------------
Unlike the simple case f(y) = -2y where λ is a fixed constant, here the
coefficient on y is -2*cos(x), which changes with x. So λ is not a single
number — it is x-dependent:

    λ(x) = -2*cos(x)

This means the "effective stiffness" of the problem varies as x advances.
At x = 0, λ = -2. At x = π/2, λ = 0. At x = π, λ = +2. The stability
of Euler's method shifts throughout the interval depending on where we are.

Stability Condition:
--------------------
For Euler's Forward method applied to dy/dx = λy, stability requires:

    |1 + h*λ| < 1

Since λ(x) = -2*cos(x), we substitute:

    |1 + h*(-2*cos(x))| < 1
    |1 - 2h*cos(x)| < 1

The worst case (most restrictive) is when |cos(x)| is maximized, i.e. cos(x) = 1
(at x = 0, 2π, 4π, ...). This gives the binding stability condition:

    |1 - 2h| < 1   →   0 < h < 1

So as long as h < 1, Euler's method is stable at the most dangerous points
in the interval. This matches what we see in the plots — h = 1.0 is right
on the boundary and starts to show visible drift.
""")

# --- Print λ and stability check at key x values ---
print(f"{'x':>8} | {'cos(x)':>10} | {'λ = -2cos(x)':>14} | {'|1-2h*λ| at h=0.5':>20}")
print("-" * 60)
for x_check in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
    lam = -2 * np.cos(x_check)
    amp = abs(1 + 0.5 * lam)
    print(f"{x_check:>8.4f} | {np.cos(x_check):>10.4f} | {lam:>14.4f} | {amp:>20.4f}")

print("""
Do the Bounding Conditions Affect the Choice of h?
---------------------------------------------------
Yes — but in a nuanced way. Because λ(x) = -2*cos(x) oscillates between
-2 and +2, the stability condition is not uniform across the interval.
The tightest constraint comes from the regions where cos(x) = 1, giving
h < 1. But near cos(x) = 0 (x ≈ π/2), the equation is essentially
non-stiff and any h would be fine locally.

In practice this means the bounding value of λ — which is |λ_max| = 2 —
is what sets the ceiling on h. We have to use a step size that keeps the
method stable at the worst-case x, even if it's overkill everywhere else.
This is why h = 0.01 and h = 0.05 perform so well: they are conservative
enough to handle the full range of λ(x) throughout [0, 10π].
""")


# In[ ]:




