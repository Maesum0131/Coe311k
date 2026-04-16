#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

t_start  = 0
t_end    = 0.01
y_0      = 0.0

h_values = [0.0005, 0.001, 0.002, 0.0025]
colors   = ['steelblue', 'darkorange', 'green', 'red']

def f(t, y):
    return -1000 * y + 3000 - 2000 * np.exp(-t)

def analytical(t):
    return 3 - 0.998 * np.exp(-1000 * t) - 2.002 * np.exp(-t)

def euler_forward(h, t_start, t_end, y_0):
    t_values = np.arange(t_start, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y_0
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
        if not np.isfinite(y_values[i]) or np.abs(y_values[i]) > 1e6:
            y_values[i:] = np.nan
            break
    return t_values, y_values

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(
    "Euler's Forward vs Analytical Solution\n"
    r"$\frac{dy}{dt} = -1000y + 3000 - 2000e^{-t},\quad y(0)=0$",
    fontsize=14, fontweight='bold'
)

t_dense = np.linspace(t_start, t_end, 5000)

# Plot full window 
ax1 = axes[0]
ax1.plot(t_dense, analytical(t_dense), 'k-', lw=2.5, label='Analytical solution', zorder=5)
for h, color in zip(h_values, colors):
    t_e, y_e = euler_forward(h, t_start, t_end, y_0)
    ax1.plot(t_e, y_e, '--o', color=color, markersize=3, alpha=0.85, label=f'Euler h={h}')
ax1.set_title("Full Range")
ax1.set_xlabel("t")
ax1.set_ylabel("y(t)")
ax1.set_ylim(-1, 4)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot Zoomed into stiff transient 
ax2 = axes[1]
t_zoom = np.linspace(0, 0.003, 2000)
ax2.plot(t_zoom, analytical(t_zoom), 'k-', lw=2.5, label='Analytical solution', zorder=5)
for h, color in zip(h_values, colors):
    t_e, y_e = euler_forward(h, t_start, 0.003, y_0)
    ax2.plot(t_e, y_e, '--o', color=color, markersize=4, alpha=0.85, label=f'Euler h={h}')
ax2.set_title("Zoomed: Stiff Transient (t = 0 to 0.003)")
ax2.set_xlabel("t")
ax2.set_ylabel("y(t)")
ax2.set_ylim(-1, 4)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("euler_stiff.png", dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:




