#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Introduction:

# This midterm applied three curve-fitting and interpolation methods natural cubic spline interpolation, 
# degree 4 polynomial fitting using least squares, and linear least squares regression to analyze U.S. GDP growth
# from 2010 to 2023. Each method revealed distinct strengths and weaknesses depending on the task. The cubic spline
# performed best for interpolation by constructing a curve that passes through every data point while maintaining 
# C0, C1, and C2 continuity at the interior knots. It also confines the impact of extreme values, such as those 
# observed during Covid, to local regions without distorting the entire curve.

# The degree 4 polynomial, is better suited for capturing long-term trends, as it smooths over pandemic outliers 
# rather than reproducing them exactly. This makes it more appropriate for macro level analysis but less reliable 
# for precise interpolation between specific quarters. The linear regression model, applied after excluding Covid 
# quarters, indicates a weak but positive long-run trend in GDP growth, thoughit simplifies the underlying dynamics.

# A key limitation across all methods is sensitivity to outliers and interpolation methods must honor extreme values 
# exactly, leading to local instability, while approximation methods spread their influence globally. For future 
# analysis, approaches such as smoothing splines or weighted least squares would provide a more robust balance by 
# allowing control over how much influence extreme values exert on the fitted model.


# In[17]:


# Part A:

# Question 1

import numpy as np
import matplotlib.pyplot as plt

# Dataset

# x is all the quarter index 1-20 (from the note it said to use for x values)
x = np.array([1, 5, 9, 13, 17, 19, 22, 25, 27, 28,
              29, 33, 37, 41, 42, 43, 45, 49, 54, 56], dtype=float) # (uses dtype float to tell numpy to have decimals) 

# y is all GDP growth value from chart
y = np.array([1.7, 0.1, 2.3, 2.7, 1.7, 5.0, 3.0, 1.5, 3.5, 1.8,
              1.3, 2.5, 3.1, -5.1, -28.1, 33.8, 6.3, -1.6, 2.4, 3.3])

# Build tridiagonal system

# Identical code from homework

def build_system(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros(n-2)
    B = np.zeros(n-2)
    C = np.zeros(n-2)
    D = np.zeros(n-2)
    for i in range(1, n-1):
        
        # variables used from homework
        
        A[i-1] = h[i-1]  # lower
        B[i-1] = 2 * (h[i-1] + h[i]) # main
        C[i-1] = h[i] # upper
        D[i-1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]) # right hand
    return A, B, C, D

# Thomas algorithm (solves the function)

# same function and works as the homework

def solve_tridiagonal(A, B, C, D):
    n = len(D)
    B = B.copy()
    D = D.copy()
    for i in range(1, n):
        factor = A[i-1] / B[i-1]
        B[i] = B[i] - factor * C[i-1]
        D[i] = D[i] - factor * D[i-1]
    M = np.zeros(n)
    M[-1] = D[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i] * M[i+1]) / B[i]
    return M

# Get second derivatives 

# boundary conditions

def get_M(x, y):
    A, B, C, D = build_system(x, y)
    M_inner = solve_tridiagonal(A, B, C, D)
    M = np.zeros(len(x))
    M[1:-1] = M_inner  # natural BC: M[0] = M[-1] = 0
    return M

# Evaluate spline at one point

# interval

def spline_value(x, y, M, x_val):
    for i in range(len(x)-1):
        if x[i] - 1e-10 <= x_val <= x[i+1] + 1e-10:
            h = x[i+1] - x[i]
            
        # cubic polynomial formula
        
            a = M[i]   * (x[i+1] - x_val)**3 / (6*h)
            b = M[i+1] * (x_val  - x[i])**3   / (6*h)
            c = (y[i]   / h - M[i]   * h/6) * (x[i+1] - x_val)
            d = (y[i+1] / h - M[i+1] * h/6) * (x_val  - x[i])
            return a + b + c + d

# Creating smooth curve

def make_spline_curve(x, y):   # creates many points for later to plot
    M = get_M(x, y)
    xs, ys = [], []
    val = x[0]
    while val <= x[-1] + 1e-10:
        result = spline_value(x, y, M, val)
        if result is not None:
            xs.append(val)
            ys.append(result)
        val += 0.01
    return xs, ys

# Generate and plot

# blue line is going to be spline
# red dots will be data points
xs, ys = make_spline_curve(x, y)

plt.figure(figsize=(12, 5))
plt.plot(xs, ys, 'b-', linewidth=2, label='Natural Cubic Spline')
plt.scatter(x, y, color='red', zorder=5, s=80, label='Data Points')

# highlight COVID quarters

covid_x = x[13:17]
covid_y = y[13:17]
plt.scatter(covid_x, covid_y, color='orange', zorder=6, s=120,
            label='COVID Quarters (2020-2021)')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Natural Cubic Spline — U.S. GDP Growth Rate (2010–2023)')
plt.xlabel('Quarter Index')
plt.ylabel('GDP Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print key values 
M = get_M(x, y)
print(f"Number of data points (n): {len(x)}")
print(f"Number of unknowns (interior M values): {len(x)-2}")
print(f"M[0] = {M[0]:.4f}  (should be 0 — natural BC)")
print(f"M[-1] = {M[-1]:.4f} (should be 0 — natural BC)")
print(f"Min spline value: {min(ys):.2f}%")
print(f"Max spline value: {max(ys):.2f}%")



# In[ ]:


# Part 1 (A)

# There are 20 data points from the table, this gives 20 second derivatives (M0 through M19). 
# The natural spline boundary conditions set the end knots or M0 and M19 to 0, which reduces the amount of unknowns
# to 18. The interior knots are to uphold three continuity conditions. The first (C0) being that the left and right
# pieces must meet at the same values, the second (C1) being the first derivatives match ensuring no sharp edges,
# and the last (C2) being the second derivatives match which ensures that the curve is smooth. The natural boundary
# conditions force the curvature to zero at both endpoints, meaning the spline transitions linearly rather than
# curving outward. 


# In[12]:


# Part 2 (A)
# Question 2
M = get_M(x, y)

x_full = np.linspace(x[0], x[-1], 1000)
y_full = []
for val in x_full:
    result = spline_value(x, y, M, val)
    if result is not None:
        y_full.append(result)
    else:
        y_full.append(np.nan)

x_full = np.array(x_full)
y_full = np.array(y_full)

covid_mask = (x >= 41) & (x <= 45)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left graph shows the norma full range
ax = axes[0]
ax.plot(x_full, y_full, 'b-', linewidth=2,
        label='Cubic Spline (full range)')
ax.scatter(x, y, color='red', zorder=5, s=80, label='20 Subset Data Points')
ax.scatter(x[covid_mask], y[covid_mask], color='orange', zorder=6,
           s=140, edgecolors='black', label='COVID Quarters (2020 Q1–2021 Q1)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Full Range (Index 1–56)')
ax.set_xlabel('Quarter Index (1 = 2010 Q1, 56 = 2023 Q4)')
ax.set_ylabel('GDP Growth Rate (%)')
ax.legend(fontsize=8)
ax.grid(True)

# Right graph is to show a zoom into index 1-30 to show spline filling gaps between sparse points

ax = axes[1]
zoom_mask = x_full <= 30
ax.plot(x_full[zoom_mask], y_full[zoom_mask], 'b-', linewidth=2,
        label='Spline (interpolated)')
ax.scatter(x[x <= 30], y[x <= 30], color='red', zorder=5, s=80,
           label='Data Points')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Zoom: Index 1–30 (Showing Gap Interpolation)')
ax.set_xlabel('Quarter Index')
ax.set_ylabel('GDP Growth Rate (%)')
ax.legend(fontsize=8)
ax.grid(True)

fig.suptitle('Question 2 — Spline Evaluated Over Full Range (2010–2023)', fontsize=13)
plt.tight_layout()
plt.show()

print(f"Spline evaluated from index {x_full[0]:.0f} to {x_full[-1]:.0f}")
print(f"NaN count (should be 0): {np.sum(np.isnan(y_full))}")
print(f"Min spline value: {np.nanmin(y_full):.2f}%")
print(f"Max spline value: {np.nanmax(y_full):.2f}%")


# In[ ]:


# Part 2 (A)
# The cubic spline is evaluated between 1000 evenly spaced points covering a index from 1 to 56 and from 2010 Q1 to
# 2023 Q4, The right panel zooms into a index 1-30 to demonstrate spline smoothly interpolates between the sparse 
# data points. Outside of the covid region the interpolant is visually smooth but between the Covid quarters the 
# spline exhibits sharp oscillation dipping and spiking much more beyond the data values. This occurs because the 
# spline must still have continuity forcing aggressive curves in neighboring intervals. This is a known limitation 
# of a cubic spline interpolation when the data set contains extreme outliers.


# Part 3 (A)

# The Runge phenomenon refers to the oscillation swinging when high degree polynomials interpolation, 
# especially when coming near the edges of the data set. Instead of a smooth curve usually the data set can 
# fluctuate in an unrealistic way. The GDP data set, the extreme values during Covid can be outliers since there 
# were other factors influencing the data. Because a cubic spline is an interpolation method, it must pass exactly 
# through all data points including the covid "extreme" values. As a result of this the spline then develops sharp
# bends to fit them. But unlike high degree polynomials, cubic splines are built from smaller functions which limit
# the oscillations to smaller local regions rather than effecting the entirety of the curve. The spline is still 
# influenced by the outliers and show noticeable curvature near the covid quarters. This highlights the trade off 
# between stability and interpolation, interpolation ensure all data points are matched and are sensitive to
# extreme values while other methods like smoothing splines or weighted least squares reduce the impact of these
# outliers and produce smooth more stable trends overall.


# In[13]:


#residual plot
M = get_M(x, y)
y_pred = np.array([spline_value(x, y, M, val) for val in x])
residuals = y - y_pred

plt.figure(figsize=(12, 4))
plt.scatter(x, residuals, color='purple', zorder=5, s=80)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Part A — Residual Plot (Spline at Knot Points)')
plt.xlabel('Quarter Index')
plt.ylabel('Residual (Actual − Predicted) %')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Residuals at each knot:")
for i in range(len(x)):
    print(f"  x={x[i]:.0f}:  residual = {residuals[i]:.2e}")


# In[14]:


# Question B


# In[15]:


# Part 1 (B)

import numpy as np
import matplotlib.pyplot as plt

# matches Part A exactly

x = np.array([1, 5, 9, 13, 17, 19, 22, 25, 27, 28,
              29, 33, 37, 41, 42, 43, 45, 49, 54, 56], dtype=float)

y = np.array([1.7, 0.1, 2.3, 2.7, 1.7, 5.0, 3.0, 1.5, 3.5, 1.8,
              1.3, 2.5, 3.1, -5.1, -28.1, 33.8, 6.3, -1.6, 2.4, 3.3])

# Vandermonde matrix

def build_vandermonde(x, degree):
    n = len(x)
    A = np.zeros((n, degree + 1))
    for i in range(n):
        for j in range(degree + 1):
            A[i, j] = x[i]**j
    return A

# Least squares 

def least_squares_poly(x, y, degree):
    A = build_vandermonde(x, degree)
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs

# Polynomial evaluation

def eval_poly(coeffs, x_vals):
    y_vals = np.zeros_like(x_vals, dtype=float)
    for j in range(len(coeffs)):
        y_vals += coeffs[j] * x_vals**j
    return y_vals

# Cubic spline copied from part A

def build_system(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros(n-2)
    B = np.zeros(n-2)
    C = np.zeros(n-2)
    D = np.zeros(n-2)
    for i in range(1, n-1):
        A[i-1] = h[i-1]
        B[i-1] = 2 * (h[i-1] + h[i])
        C[i-1] = h[i]
        D[i-1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    return A, B, C, D

def solve_tridiagonal(A, B, C, D):
    n = len(D)
    B = B.copy()
    D = D.copy()
    for i in range(1, n):
        factor = A[i-1] / B[i-1]
        B[i] -= factor * C[i-1]
        D[i] -= factor * D[i-1]
    M = np.zeros(n)
    M[-1] = D[-1] / B[-1]
    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i] * M[i+1]) / B[i]
    return M

def get_M(x, y):
    A, B, C, D = build_system(x, y)
    M_inner = solve_tridiagonal(A, B, C, D)
    M = np.zeros(len(x))
    M[1:-1] = M_inner
    return M

def spline_value(x, y, M, x_val):
    for i in range(len(x)-1):
        if x[i] - 1e-10 <= x_val <= x[i+1] + 1e-10:
            h = x[i+1] - x[i]
            a = M[i]   * (x[i+1] - x_val)**3 / (6*h)
            b = M[i+1] * (x_val  - x[i])**3   / (6*h)
            c = (y[i]   / h - M[i]   * h/6) * (x[i+1] - x_val)
            d = (y[i+1] / h - M[i+1] * h/6) * (x_val  - x[i])
            return a + b + c + d

# Condition check

degree = 4
A_vand = build_vandermonde(x, degree)
cond = np.linalg.cond(A_vand.T @ A_vand)
print(f"Condition number of ATA: {cond:.2e}")
if cond > 1e12:
    print("WARNING: poorly conditioned — results may be inaccurate")
else:
    print("Condition number is acceptable")

# Four degree polynomial fit

coeffs = least_squares_poly(x, y, degree)
print(f"\nPolynomial Coefficients (degree 0 to 4):")
for i, c in enumerate(coeffs):
    print(f"  c{i} = {c:.6f}")

# Generating smooth curves 

x_dense = np.linspace(x[0], x[-1], 500)

y_poly = eval_poly(coeffs, x_dense)

M = get_M(x, y)
y_spline = np.array([spline_value(x, y, M, val) for val in x_dense])

# Plotting spline vs polynomial

plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_spline, 'b-', linewidth=2, label='Cubic Spline (Part A)')
plt.plot(x_dense, y_poly, 'g--', linewidth=2, label='Degree-4 Polynomial (Least Squares)')
plt.scatter(x, y, color='red', zorder=5, s=80, label='Data Points')
covid_mask = (x >= 41) & (x <= 45)
plt.scatter(x[covid_mask], y[covid_mask], color='orange', zorder=6,
            s=140, edgecolors='black', label='COVID Quarters')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Part B – Q1: Cubic Spline vs Degree-4 Polynomial Fit')
plt.xlabel('Quarter Index (1 = 2010 Q1, 56 = 2023 Q4)')
plt.ylabel('GDP Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual plot

y_pred = eval_poly(coeffs, x)
residuals = y - y_pred

plt.figure(figsize=(10, 4))
plt.scatter(x, residuals, color='purple', zorder=5, s=80)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title('Part B – Q1: Residual Plot (Degree-4 Polynomial)')
plt.xlabel('Quarter Index')
plt.ylabel('Residual (Actual − Predicted) %')
plt.grid(True)
plt.tight_layout()
plt.show()

rmse = np.sqrt(np.mean(residuals**2))
print(f"\nRMSE (Degree-4 Polynomial): {rmse:.4f}%")


# In[ ]:


# Part 1 (B)

# The cubic spline and degree 4 polynomial both model the GDP data but have different functions and purposes. 
# The cubic spline passes through all 20 data points and captures local variations including the sharp Covid 19 
# swing from -28.1% to 33.8%. However this exact interpolation makes it sensitive to extreme outliers, the covid
# values  force aggressive curvature in neighboring intervals. The degree 4 polynomial does not pass through every 
# point but better captures the overall long-run trend by spreading the influence of extreme values across the
# entire fit. Near the Covid quarters the polynomial smooths over the spike rather than reproducing it exactly, 
# which is more economically meaningful for trend analysis. In conclusion the cubic spline better reproduces
# specific data points while the degree 4 polynomial better captures the overall trend.  


# In[18]:


# Part 2 (B)

import numpy as np
import matplotlib.pyplot as plt

# Same dataset as before
x = np.array([1, 5, 9, 13, 17, 19, 22, 25, 27, 28,
              29, 33, 37, 41, 42, 43, 45, 49, 54, 56], dtype=float)

y = np.array([1.7, 0.1, 2.3, 2.7, 1.7, 5.0, 3.0, 1.5, 3.5, 1.8,
              1.3, 2.5, 3.1, -5.1, -28.1, 33.8, 6.3, -1.6, 2.4, 3.3])

# COVID quarters are indices 41, 42, 43, 45 in x (2020 Q1 through 2021 Q1)
# Create a mask to exclude them
covid_mask = (x >= 41) & (x <= 45)
x_no_covid = x[~covid_mask]
y_no_covid = y[~covid_mask]

print(f"Full dataset: {len(x)} points")
print(f"After removing COVID quarters: {len(x_no_covid)} points")
print(f"Removed x indices: {x[covid_mask]}")
print(f"Removed y values: {y[covid_mask]}")

# Build Vandermonde matrix for degree 1 (linear)
def build_vandermonde(x, degree):
    n = len(x)
    A = np.zeros((n, degree + 1))
    for i in range(n):
        for j in range(degree + 1):
            A[i, j] = x[i]**j
    return A

# Solve normal equations: (A^T A) coeffs = A^T y
def least_squares_poly(x, y, degree):
    A = build_vandermonde(x, degree)
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs

# Evaluate polynomial
def eval_poly(coeffs, x_vals):
    y_vals = np.zeros_like(x_vals, dtype=float)
    for j in range(len(coeffs)):
        y_vals += coeffs[j] * x_vals**j
    return y_vals

# Fit degree-1 (linear) model to non-COVID data
coeffs_linear = least_squares_poly(x_no_covid, y_no_covid, degree=1)
intercept = coeffs_linear[0]
slope     = coeffs_linear[1]

print(f"\nLinear Least Squares (COVID excluded):")
print(f"  Intercept (c0): {intercept:.4f}%")
print(f"  Slope     (c1): {slope:.6f}% per quarter index")
print(f"  Interpreted: GDP grows approximately {slope:.4f}% per quarter index unit")

# Generate smooth line over full range for plotting
x_dense = np.linspace(x[0], x[-1], 500)
y_linear_full = eval_poly(coeffs_linear, x_dense)

# Residuals on non-COVID points only
y_pred_no_covid = eval_poly(coeffs_linear, x_no_covid)
residuals = y_no_covid - y_pred_no_covid
rmse = np.sqrt(np.mean(residuals**2))
print(f"  RMSE (non-COVID points): {rmse:.4f}%")

# --- Plot 1: Linear fit over the data ---
plt.figure(figsize=(12, 5))
plt.plot(x_dense, y_linear_full, 'g-', linewidth=2, label='Linear Least Squares Fit (COVID excluded)')
plt.scatter(x_no_covid, y_no_covid, color='red', zorder=5, s=80, label='Non-COVID Data Points')
plt.scatter(x[covid_mask], y[covid_mask], color='orange', zorder=6, s=120,
            edgecolors='black', label='COVID Quarters (excluded from fit)')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Part B – Q2: Linear Least Squares Fit (COVID Quarters Excluded)')
plt.xlabel('Quarter Index (1 = 2010 Q1, 56 = 2023 Q4)')
plt.ylabel('GDP Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Residual plot ---
plt.figure(figsize=(12, 4))
plt.scatter(x_no_covid, residuals, color='purple', zorder=5, s=80)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title('Part B – Q2: Residual Plot (Linear Fit, COVID Excluded)')
plt.xlabel('Quarter Index')
plt.ylabel('Residual (Actual − Predicted) %')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[19]:


# Part 2 (B)

# The linear least squares model was fit for 16 points after excluding the 4 covid points. The resulting slope is 
# approximately -0.001907% per quarter index which indicates a slight negative long run trend in the GDP growth 
# over the period when Covid was removed. Whether the linear trends makes economic sense depends on the assumption 
# it requires. A linear model assumes that the GDP growth changes by a constant amount each quarter meaning a 
# growth either steadily or decelerates over a fixed rate of time. GDP growth rate is cyclical and mean reverting: 
# expansions and contractions follow each other in irregular patterns driven my monetary policy and external factors
#. The low RMSE suggest the linear model is a reasonable approximation of the trend, but the residual plot reveals 
# structures as the points ate not randomly scattered around zero, which indicates that the true relationship has 
# nonlinear components that the model cannot capture.


# In[20]:


# Question C

# Smoothness

# The cubic spline guarantees C0, C1, and C2 continuity at every interior knot, meaning the interpolated curve has 
# no sharp corners or discontinous anywhere in the range. This matters for economic interpretation because a 
# GDP estimate that jumps between quarters can be misleading to a policymaker trying to understand gradual economic
# momentum. Furthermore, C² continuity ensures that not only GDP growth, but also the rate of change of growth 
# or economic acceleration, varies smoothly—better reflecting real economic behavior. The degree-4 polynomial is
# also smooth globally, but its smoothness comes at the cost of fitting the data less faithfully, since it does not
# pass through every known quarter.


# Accuracy

# Because the cubic spline is an interpolant, it passes exactly through all 20 known data points. For a policymaker
# estimating a value between two specific known quarters, the spline guarantees that the surrounding anchor points
# are honored precisely. In contrast, the degree-4 polynomial minimizes overall squared error across all points, 
# meaning individual data points may deviate from their true values. The spline achieves zero residual at all knot
# points by construction, ensuring exact agreement with observed GDP values, whereas the polynomial introduces 
# approximation error even at measured quarters. For interpolation tasks, this makes the spline strictly more 
# reliable.

# Sensitivity

# The COVID quarters like 2020 Q2 with −28.1 and 2020 Q3 with +33.8% present a major challenge for both methods. 
# The cubic spline is locally sensitive—these extreme values produce strong curvature, but only in neighboring 
# intervals, leaving the rest of the curve largely unaffected. In contrast, the degree 4 polynomial is globally 
# sensitive because it fits a single function across all data points, the COVID outliers distort the curve across
# the entire range, even for distant quarters. However, the spline’s exact interpolation also introduces a 
# limitation as these extreme values can create unrealistic curvature locally, ultimately highlighting a trade-off between 
# exactness and stability.

# Quantitative Support

# The spline achieves zero residual at all knot points by construction, while the degree 4 polynomial produces a 
# non zero RMSE across the dataset. This confirms that the spline is exact at observed values whereas the 
# polynomial sacrifices pointwise accuracy for a smoother global fit. Additionally the condition number of the 
# Vandermonde system used for the polynomial fit was verified to be within an acceptable range, indicating that 
# numerical instability is not the primary issue the difference in performance is due to the modeling approach 
# itself.

# Big O Consideration

# Both methods are computationally efficient for datasets of this size. The cubic spline requires solving a
# tridiagonal system in O(n) time using the Thomas algorithm, while the polynomial least squares method requires
# solving the normal equations with complexity O(n·d^2). Although the spline scales more efficiently for larger 
# datasets, computational cost is not a deciding factor here; the recommendation is driven by accuracy and stability
# considerations.

# My reccomadtion

# Therefore, the cubic spline is the most appropriate method for interpolation tasks requiring accurate estimates 
# between known data points. It preserves local structure, guarantees exact agreement with observed values, and 
# limits the influence of outliers to nearby intervals. In contrast, polynomial and linear least squares models are
# better suited for identifying long-term trends, where capturing overall behavior is more important than exact 
# local accuracy.


# In[ ]:


# Conclusion

# This project applied cubic spline interpolation, degree-4 polynomial fitting, and linear least squares regression
# to U.S. GDP growth data, highlighting the trade offs between interpolation and approximation methods.

# The cubic spline performed best for interpolation, it passes exactly through all data points while maintaining
# smoothness and limiting the impact of extreme Covid values to nearby intervals. The degree 4 polynomial better
# captured the overall trend by smoothing short-term fluctuations, while the linear model, excluding Covid quarters,
# showed a weak positive long-term trend but oversimplifies the data.

# Overall, the results demonstrate that method selection depends on the objective of the splines are best for
# accurate interpolation, while polynomial and linear models are better suited for trend analysis. All methods 
# remain sensitive to extreme values, emphasizing the importance of choosing models based on both the data and the 
# intended application.

