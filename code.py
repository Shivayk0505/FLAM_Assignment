import numpy as np
import pandas as pd
from math import radians
from scipy.optimize import least_squares

# --- load your data ---
data_path = "/home/user/Flam/xy_data.csv"  # adjust path as needed
df = pd.read_csv(data_path)  # columns: x,y
xy = df[['x','y']].to_numpy()

def residuals(p, xy):
    theta, M, X = p
    c, s = np.cos(theta), np.sin(theta)
    x, y = xy[:,0], xy[:,1]
    xp = (x - X)*c + (y - 42.0)*s
    yp = -(x - X)*s + (y - 42.0)*c
    yhat = np.exp(M*np.abs(xp)) * np.sin(0.3*xp)
    return yp - yhat

# bounds (theta in radians)
lo = np.array([radians(0.0),   -0.05,   0.0])
hi = np.array([radians(50.0),   0.05, 100.0])

# sensible initial guess
p0 = np.array([radians(20.0), 0.0, 10.0])

sol = least_squares(
    residuals, p0, args=(xy,),
    bounds=(lo, hi),
    loss='soft_l1', f_scale=0.5, max_nfev=20000
)

theta, M, X = sol.x
print(f"theta (deg) = {np.degrees(theta):.6f}")
print(f"M           = {M:.6f}")
print(f"X           = {X:.6f}")

# Optional quality report
r = residuals(sol.x, xy)
print(f"L1 mean |res| = {np.mean(np.abs(r)):.6f}, L2 RMSE = {np.sqrt(np.mean(r**2)):.6f}")

# Desmos / LaTeX-friendly outputs
print("\nDesmos parametric (copy paste):")
print(f"x(t) = t*cos({theta}) - exp({M}*abs(t))*sin(0.3*t)*sin({theta}) + {X}")
print(f"y(t) = 42 + t*sin({theta}) + exp({M}*abs(t))*sin(0.3*t)*cos({theta})")

# If you want to compute the assignment's L1 curve distance on a uniform t-grid:
def param_curve(t, theta, M, X):
    x = t*np.cos(theta) - np.exp(M*np.abs(t))*np.sin(0.3*t)*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + np.exp(M*np.abs(t))*np.sin(0.3*t)*np.cos(theta)
    return np.stack([x,y], axis=1)

# uniform t sampling for 6 < t < 60
t_grid = np.linspace(6, 60, 500)
curve = param_curve(t_grid, theta, M, X)

# crude L1 distance: each data point to nearest curve sample
from scipy.spatial import cKDTree
tree = cKDTree(curve)
dists, _ = tree.query(xy, k=1)
print(f"\nApprox L1 distance sum to sampled curve = {np.sum(np.abs(dists)):.6f}")
