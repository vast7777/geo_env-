import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

# paths
shapefile_path = r"C:\Users\15837\geo_env-\WS_3\WS_3.shp"
data_dir = r"C:\Users\15837\geo_env-\assignment7\OneDrive_1_2026-3-11"

gdf = gpd.read_file(shapefile_path)
print("Watershed shapefile loaded")

# load and clip ERA5 data to watershed
def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    ds = ds.rio.write_crs("EPSG:4326")
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped[var_name]

# Part 1 load 2001 data
print("\nLoading 2001 data...")
P_grid = load_and_clip(os.path.join(data_dir, "Precipitation", "era5_OLR_2001_total_precipitation.nc"), "tp", gdf) * 1000
ET_grid = load_and_clip(os.path.join(data_dir, "Total_Evaporation", "era5_OLR_2001_total_evaporation.nc"), "e", gdf) * 1000
Q_grid = load_and_clip(os.path.join(data_dir, "Runoff", "ambientera5_OLR_2001_total_runoff.nc"), "ro", gdf) * 1000

# watershed averaged values
P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

# evaporation should be positive
ET = np.where(ET < 0, -ET, ET)

print(f"2001 data: {len(P)} hours")
print(f"  P mean: {np.mean(P):.4f} mm/h, total: {np.sum(P):.1f} mm")
print(f"  ET mean: {np.mean(ET):.4f} mm/h, total: {np.sum(ET):.1f} mm")
print(f"  Q mean: {np.mean(Q_obs):.4f} mm/h, total: {np.sum(Q_obs):.1f} mm")

# plot 2001 variables
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
hours = np.arange(len(P))

axes[0].plot(hours, P, color='steelblue', linewidth=0.5)
axes[0].set_ylabel('Precipitation (mm/h)')
axes[0].set_title('Hourly Hydrological Variables - 2001')

axes[1].plot(hours, ET, color='orangered', linewidth=0.5)
axes[1].set_ylabel('Evaporation (mm/h)')

axes[2].plot(hours, Q_obs, color='seagreen', linewidth=0.5)
axes[2].set_ylabel('Runoff (mm/h)')
axes[2].set_xlabel('Time (hour)')

plt.tight_layout()
plt.savefig('a8_variables_2001.png', dpi=200)
plt.close()
print("Saved: a8_variables_2001.png")

# load 2002 data
print("\nLoading 2002 data...")
P_grid_v = load_and_clip(os.path.join(data_dir, "Precipitation", "era5_OLR_2002_total_precipitation.nc"), "tp", gdf) * 1000
ET_grid_v = load_and_clip(os.path.join(data_dir, "Total_Evaporation", "era5_OLR_2002_total_evaporation.nc"), "e", gdf) * 1000
Q_grid_v = load_and_clip(os.path.join(data_dir, "Runoff", "ambientera5_OLR_2002_total_runoff.nc"), "ro", gdf) * 1000

P_v = P_grid_v.mean(dim=["latitude", "longitude"]).values
ET_v = ET_grid_v.mean(dim=["latitude", "longitude"]).values
Q_obs_v = Q_grid_v.mean(dim=["latitude", "longitude"]).values
ET_v = np.where(ET_v < 0, -ET_v, ET_v)

print(f"2002 data: {len(P_v)} hours")

# plot 2002 variables
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
hours_v = np.arange(len(P_v))

axes[0].plot(hours_v, P_v, color='steelblue', linewidth=0.5)
axes[0].set_ylabel('Precipitation (mm/h)')
axes[0].set_title('Hourly Hydrological Variables - 2002')

axes[1].plot(hours_v, ET_v, color='orangered', linewidth=0.5)
axes[1].set_ylabel('Evaporation (mm/h)')

axes[2].plot(hours_v, Q_obs_v, color='seagreen', linewidth=0.5)
axes[2].set_ylabel('Runoff (mm/h)')
axes[2].set_xlabel('Time (hour)')

plt.tight_layout()
plt.savefig('a8_variables_2002.png', dpi=200)
plt.close()
print("Saved: a8_variables_2002.png")

# linear reservoir model
def simulate_runoff(k, P, ET, Q_init, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q_init
    for t in range(1, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt / k)
        Q_sim[t] = max(0, Q_t)
    return Q_sim

# KGE metric
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    KGE = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE, r, alpha, beta

# Part 2 validate with given k
print("\n--- Part 2: Validation with k=0.15 ---")
k_test = 0.15
Q_sim_val = simulate_runoff(k_test, P, ET, Q_obs[0])
KGE_val, r_val, alpha_val, beta_val = kge(Q_obs, Q_sim_val)

print(f"KGE:         {KGE_val:.4f}  (ideal: 1)")
print(f"Correlation: {r_val:.4f}  (ideal: 1)")
print(f"Alpha:       {alpha_val:.4f}  (ideal: 1)")
print(f"Beta:        {beta_val:.4f}  (ideal: 1)")

# time series plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hours, Q_obs, label='Observed (ERA5)', color='steelblue', linewidth=0.5)
ax.plot(hours, Q_sim_val, label='Simulated (k=0.15)', color='orangered', linewidth=0.5)
ax.set_xlabel('Time (hour)')
ax.set_ylabel('Runoff (mm/h)')
ax.set_title(f'Validation: Observed vs Simulated Runoff (2001, k=0.15, KGE={KGE_val:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('a8_validation_ts.png', dpi=200)
plt.close()
print("Saved: a8_validation_ts.png")

# scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(Q_obs, Q_sim_val, s=2, alpha=0.3)
max_val = max(np.max(Q_obs), np.max(Q_sim_val))
ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax.set_xlabel('Observed Runoff (mm/h)')
ax.set_ylabel('Simulated Runoff (mm/h)')
ax.set_title('Scatter: Observed vs Simulated (Validation, k=0.15)')
ax.legend()
plt.tight_layout()
plt.savefig('a8_validation_scatter.png', dpi=200)
plt.close()
print("Saved: a8_validation_scatter.png")

# Part 3 calibration
print("\n--- Part 3: Calibration ---")

def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET, Q_obs[0])
    kge_val, _, _, _ = kge(Q_obs, Q_sim)
    return 1 - kge_val

res = opt.minimize_scalar(objective, bounds=(0.01, 5), args=(P, ET, Q_obs), method='bounded')
best_k = res.x
print(f"Optimized k: {best_k:.4f}")

Q_sim_cal = simulate_runoff(best_k, P, ET, Q_obs[0])
KGE_cal, r_cal, alpha_cal, beta_cal = kge(Q_obs, Q_sim_cal)

print(f"KGE:         {KGE_cal:.4f}  (ideal: 1)")
print(f"Correlation: {r_cal:.4f}  (ideal: 1)")
print(f"Alpha:       {alpha_cal:.4f}  (ideal: 1)")
print(f"Beta:        {beta_cal:.4f}  (ideal: 1)")

# calibration time series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hours, Q_obs, label='Observed (ERA5)', color='steelblue', linewidth=0.5)
ax.plot(hours, Q_sim_cal, label=f'Simulated (k={best_k:.3f})', color='orangered', linewidth=0.5)
ax.set_xlabel('Time (hour)')
ax.set_ylabel('Runoff (mm/h)')
ax.set_title(f'Calibration: Observed vs Simulated Runoff (2001, k={best_k:.3f}, KGE={KGE_cal:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('a8_calibration_ts.png', dpi=200)
plt.close()
print("Saved: a8_calibration_ts.png")

# calibration scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(Q_obs, Q_sim_cal, s=2, alpha=0.3)
max_val = max(np.max(Q_obs), np.max(Q_sim_cal))
ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax.set_xlabel('Observed Runoff (mm/h)')
ax.set_ylabel('Simulated Runoff (mm/h)')
ax.set_title(f'Scatter: Observed vs Simulated (Calibration, k={best_k:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('a8_calibration_scatter.png', dpi=200)
plt.close()
print("Saved: a8_calibration_scatter.png")

# validation on 2002 with calibrated k
print(f"\n--- Part 3: Validation on 2002 with calibrated k={best_k:.4f} ---")
Q_sim_v = simulate_runoff(best_k, P_v, ET_v, Q_obs_v[0])
KGE_v, r_v, alpha_v, beta_v = kge(Q_obs_v, Q_sim_v)

print(f"KGE:         {KGE_v:.4f}  (ideal: 1)")
print(f"Correlation: {r_v:.4f}  (ideal: 1)")
print(f"Alpha:       {alpha_v:.4f}  (ideal: 1)")
print(f"Beta:        {beta_v:.4f}  (ideal: 1)")

# validation 2002 time series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hours_v, Q_obs_v, label='Observed (ERA5)', color='steelblue', linewidth=0.5)
ax.plot(hours_v, Q_sim_v, label=f'Simulated (k={best_k:.3f})', color='orangered', linewidth=0.5)
ax.set_xlabel('Time (hour)')
ax.set_ylabel('Runoff (mm/h)')
ax.set_title(f'Validation 2002: Observed vs Simulated Runoff (k={best_k:.3f}, KGE={KGE_v:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('a8_validation2002_ts.png', dpi=200)
plt.close()
print("Saved: a8_validation2002_ts.png")

# validation 2002 scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(Q_obs_v, Q_sim_v, s=2, alpha=0.3)
max_val = max(np.max(Q_obs_v), np.max(Q_sim_v))
ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
ax.set_xlabel('Observed Runoff (mm/h)')
ax.set_ylabel('Simulated Runoff (mm/h)')
ax.set_title(f'Scatter: Observed vs Simulated (Validation 2002, k={best_k:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('a8_validation2002_scatter.png', dpi=200)
plt.close()
print("Saved: a8_validation2002_scatter.png")

print("\nAll done!")
