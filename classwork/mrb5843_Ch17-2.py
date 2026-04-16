#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
 
x_start, x_end = 0, 10
step_sizes = [0.1, 0.5, 1.0, 1.5]
colors = ['steelblue', 'darkorange', 'green', 'red']
 
def f(x, y):
    return -2 * y
 
def exact_solution(x):
    return np.exp(-2 * x)
 

 
def build_euler_matrix(h):
    """Construct the Euler amplification matrix for dy/dx = -2y."""
    A = np.array([[1 - 2 * h]])
    return A
 
 
print("=" * 60)
print("EIGENVALUE STABILITY ANALYSIS")
print("=" * 60)
print("""
For dy/dx = -2y, one step of Euler's method gives:
  y_{n+1} = (1 - 2h) * y_n
 
The amplification matrix A = [[1 - 2h]] has eigenvalue λ = 1 - 2h.
If |λ| < 1, each step shrinks the solution — that's stable.
If |λ| ≥ 1, the solution grows or oscillates without bound — unstable.
""")
 
stability_results = {}
 
for h_val in step_sizes:
    A = build_euler_matrix(h_val)
    eigenvalues = np.linalg.eigvals(A)   # built-in eigenvalue function
    max_magnitude = np.max(np.abs(eigenvalues))
 
    print(f"  Step size h = {h_val}")
    print(f"    Eigenvalue λ = {eigenvalues[0]:.4f}   |λ| = {max_magnitude:.4f}")
 
    if max_magnitude < 1:
        status = "stable"
        print("    → Solution is Stable\n")
    else:
        status = "unstable"
        print("    → Solution may not be Entirely Stable ;)\n")
 
    stability_results[h_val] = (max_magnitude, status)
 



 
def compute_errors(h_val):
    """Run Euler's method and compute local + global error vs exact."""
    n = int((x_end - x_start) / h_val)
    x_vals = np.linspace(x_start, x_end, n + 1)
    y_euler = np.zeros(n + 1)
    y_euler[0] = 1.0
    for i in range(n):
        y_euler[i + 1] = y_euler[i] + h_val * f(x_vals[i], y_euler[i])
    y_exact = exact_solution(x_vals)
    local_error = np.abs(y_euler - y_exact)
    global_error = np.mean(local_error)
    return x_vals, y_euler, y_exact, local_error, global_error
 
 
 
 
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle("Euler's Method: dy/dx = -2y — Eigenvalue Stability Analysis",
             fontsize=15, fontweight='bold')
 

ax1 = axes[0]
x_dense = np.linspace(x_start, x_end, 1000)
ax1.plot(x_dense, exact_solution(x_dense), 'k-', lw=2.5,
         label='Exact: $e^{-2x}$', zorder=5)
 
for h_val, color in zip(step_sizes, colors):
    x_v, y_e, _, _, _ = compute_errors(h_val)
    mag, status = stability_results[h_val]
    tag = "Stable" if status == "stable" else "Unstable"
    ax1.plot(x_v, y_e, '--o', color=color, markersize=3, alpha=0.85,
             label=f'h={h_val}  |λ|={mag:.2f}  [{tag}]')
 
ax1.set_title("Euler Approximation vs Exact Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim(-2, 2)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', lw=0.8)
 

ax2 = axes[1]
global_errors = [compute_errors(h_val)[4] for h_val in step_sizes]
bar_colors = ['steelblue' if stability_results[h][1] == 'stable' else 'red'
              for h in step_sizes]
bars = ax2.bar([str(h) for h in step_sizes], global_errors,
               color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for bar, err in zip(bars, global_errors):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{err:.4f}', ha='center', va='bottom', fontsize=9)
ax2.set_title("Global (Average) Error vs Step Size  [blue = stable, red = unstable]")
ax2.set_xlabel("Step Size h")
ax2.set_ylabel("Mean |Euler − Exact|")
ax2.grid(True, alpha=0.3, axis='y')
 

ax3 = axes[2]
for h_val, color in zip(step_sizes, colors):
    x_v, _, _, l_err, _ = compute_errors(h_val)
    mag, status = stability_results[h_val]
    tag = "Stable" if status == "stable" else "Unstable"
    ax3.plot(x_v, l_err, '-', color=color, alpha=0.85,
             label=f'h={h_val}  [{tag}]')
ax3.set_title("Local Error at Each x  |Euler − Exact|")
ax3.set_xlabel("x")
ax3.set_ylabel("Local Error")
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("euler_eigenvalue.png", dpi=150, bbox_inches='tight')
plt.show()
 


# In[7]:


# Written explanations for each question

 
print("=" * 60)
print("Q1: What is the coefficient matrix and why?")
print("=" * 60)
print("""
For dy/dx = -2y, Euler's update rule is y_{n+1} = y_n + h*(-2)*y_n,
which simplifies to y_{n+1} = (1 - 2h) * y_n. The coefficient matrix
is just the scalar A = [[1 - 2h]] — it represents how much the solution
is scaled at every step. If that scaling factor is less than 1 in
magnitude, the solution decays toward zero as expected. If it exceeds 1,
the method amplifies the solution instead of damping it.
""")
 
print("=" * 60)
print("Q2: What do the eigenvalues tell us?")
print("=" * 60)
print("""
The eigenvalue of the amplification matrix is λ = 1 - 2h. This single
number tells the whole stability story. At h = 0.1, λ = 0.8 — safely
inside the unit circle, so the method is stable. At h = 1.0, λ = -1.0 —
right on the boundary, leading to sign-alternating oscillations. At
h = 1.5, λ = -2.0 — outside the unit circle entirely, meaning the method
amplifies the error at every step and the solution diverges. Eigenvalue
magnitude is a clean, mathematical way to diagnose instability before
even running the simulation.
""")
 
print("=" * 60)
print("Q3: What does the simulation confirm?")
print("=" * 60)
print("""
The plots confirm exactly what the eigenvalue analysis predicts. h = 0.1
and h = 0.5 track the exact solution well, with small and growing but
bounded errors. h = 1.0 starts oscillating around zero — the eigenvalue
of -1 causes the solution to flip sign every step. h = 1.5 diverges
completely, exploding away from the true curve almost immediately. The
eigenvalue check gives us the answer analytically; the simulation just
makes it visual.
""")
 
print("=" * 60)
print("Q4: Key takeaway on Euler stability")
print("=" * 60)
print("""
Euler's method for dy/dx = -2y is only stable when |1 - 2h| < 1, which
requires 0 < h < 1. Beyond h = 1, the method is guaranteed to be
unstable regardless of how smooth the exact solution is. This is the
fundamental limitation of explicit Euler — stability imposes a strict
ceiling on step size that is entirely separate from accuracy. For stiff
problems with large coefficients (like -20y instead of -2y), that ceiling
drops even lower, making Euler impractical without a very tiny h.
""")
 


# In[ ]:




