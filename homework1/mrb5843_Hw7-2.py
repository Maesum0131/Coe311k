#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_start = 0
x_end = 10 * np.pi
y0 = 1  # y(0) = 1

# ODE: dy/dx = -2y*cos(x)
def f(x, y):
    return -2 * y * np.cos(x)

# Exact solution y(x) = e^(-2*sin(x))
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
# Plot 

fig, axes = plt.subplots(3, 1, figsize=(13, 16))
fig.suptitle("Euler's Method: dy/dx = -2y·cos(x), y(0)=1",
             fontsize=15, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 1, len(step_sizes)))

#  Euler solution for each h vs exact 
ax1 = axes[0]
x_exact_dense = np.linspace(x_start, x_end, 5000)
ax1.plot(x_exact_dense, exact_solution(x_exact_dense), 'k-', lw=2,
         label='Exact solution', zorder=5)
for h, color in zip(step_sizes, colors):
    x_vals, y_vals = euler_method(h)
    ax1.plot(x_vals, y_vals, '-', color=color, lw=1.2,
             label=f'h={h}', alpha=0.8)
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

#  Summary table 
print(f"\n{'h':>6} | {'Avg y':>10} | {'Avg Error':>12} | {'Local Error @ x≈15':>20}")
print("-" * 58)
for h, avg_y, avg_err, loc_err in zip(step_sizes, avg_y_values, avg_errors, local_errors_at_15):
    print(f"{h:>6} | {avg_y:>10.5f} | {avg_err:>12.6f} | {loc_err:>20.6f}")


# Written explanation

print("""
================================================================================
WHAT WE'RE SEEING AND WHY
================================================================================

Plot 1 — Euler's Method vs Exact Solution:
The exact solution y(x) = e^(-2*sin(x)) is a smooth, bounded oscillation that
stays between roughly e^(-2) ≈ 0.135 and e^2 ≈ 7.39. Small step sizes like
h = 0.01 and h = 0.05 hug the exact curve closely throughout the entire
interval. As h grows, the approximation starts drifting — especially near the
peaks and troughs where the function changes most rapidly. By h = 0.75 and
h = 1.0, the Euler solution has accumulated enough error that it visibly
diverges from the true curve, particularly over the long [0, 10π] interval.

Plot 2 — Average Error vs h:
The average error grows monotonically with step size, which is exactly what
we expect from a first-order method like Euler's. The relationship is roughly
linear — doubling h approximately doubles the error. This is the defining
characteristic of O(h) convergence. Small h means more steps and more
computation, but the payoff is a solution that stays close to the truth across
the entire domain.

Plot 3 — Local Error at x ≈ 15:
The local error at x ≈ 15 tells a similar story but for a single specific
point. The error grows with h, but not perfectly smoothly — this is because
x = 15 sits near a peak of sin(x), where the solution is curving sharply and
Euler's linear approximation struggles most. Larger step sizes overshoot or
undershoot the curve at exactly the worst possible moments, amplifying the
local error disproportionately compared to flatter regions of the domain.

Key Takeaway:
Euler's method is simple and intuitive, but it pays a price for that
simplicity. Over a long interval like [0, 10π], errors accumulate with every
step. The only lever we have is step size — smaller h gives better accuracy
but at higher computational cost. For oscillatory problems like this one,
where the solution swings up and down repeatedly, that error accumulation is
especially visible because each new oscillation inherits whatever drift built
up in the previous one.
""")


# In[ ]:




