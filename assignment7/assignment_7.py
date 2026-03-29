
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import mapping
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# data paths
base_dir = r"C:\Users\15837\geo_env-\assignment7"
precip_dir = os.path.join(base_dir, 'OneDrive_1_2026-3-11', 'Precipitation')
evap_dir = os.path.join(base_dir, 'OneDrive_1_2026-3-11', 'Total_Evaporation')
runoff_dir = os.path.join(base_dir, 'OneDrive_1_2026-3-11', 'Runoff')
shp_path = os.path.join(base_dir, 'saudiSHP', 'Saudi_Shape.shp')

# load shapefile
gdf = gpd.read_file(shp_path)
print("Shapefile loaded:", len(gdf), "features")
print("Columns:", list(gdf.columns))

# load all nc files for a given variable folder
def load_files(folder):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')])
    print(f"  Loading {len(files)} files from {os.path.basename(folder)}")
    datasets = [xr.open_dataset(f) for f in files]
    ds = xr.concat(datasets, dim='valid_time')
    return ds

print("\nLoading data...")
ds_tp = load_files(precip_dir)
ds_e = load_files(evap_dir)
ds_ro = load_files(runoff_dir)

# get variable names from each dataset
tp_var = list(ds_tp.data_vars)[0]
e_var = list(ds_e.data_vars)[0]
ro_var = list(ds_ro.data_vars)[0]
print(f"Variables: tp={tp_var}, e={e_var}, ro={ro_var}")

# figure out time and spatial dimension names
time_dim = 'valid_time' if 'valid_time' in ds_tp[tp_var].dims else 'time'
spatial_dims = [d for d in ds_tp[tp_var].dims if d != time_dim]

# area weighted spatial mean using cosine of latitude
weights = np.cos(np.deg2rad(ds_tp['latitude']))

def spatial_mean(da):
    return (da * weights).mean(dim=spatial_dims) / weights.mean()

# resample hourly data to monthly sums, convert m to mm
tp_monthly = spatial_mean(ds_tp[tp_var]).resample({time_dim: 'ME'}).sum() * 1000
e_monthly = -spatial_mean(ds_e[e_var]).resample({time_dim: 'ME'}).sum() * 1000  # ERA5 evaporation is negative
ro_monthly = spatial_mean(ds_ro[ro_var]).resample({time_dim: 'ME'}).sum() * 1000

time = tp_monthly[time_dim].values
time_mpl = mdates.date2num(pd.to_datetime(time))  # convert to matplotlib date numbers for bar plots

print(f"\nTime range: {time[0]} to {time[-1]}")
print(f"Total months: {len(time)}")
print(f"Mean monthly P: {float(tp_monthly.mean()):.2f} mm")
print(f"Mean monthly E: {float(e_monthly.mean()):.2f} mm")
print(f"Mean monthly R: {float(ro_monthly.mean()):.2f} mm")

# plot precipitation
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(time_mpl, tp_monthly.values, width=25, color='steelblue', alpha=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('Precipitation (mm/month)')
ax.set_title('Monthly Total Precipitation over Saudi Arabia (2000-2020)')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_precipitation.png', dpi=200)
plt.close()
print("Saved: a7_precipitation.png")

# plot evaporation
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(time_mpl, e_monthly.values, width=25, color='orangered', alpha=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('Evaporation (mm/month)')
ax.set_title('Monthly Total Evaporation over Saudi Arabia (2000-2020)')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_evaporation.png', dpi=200)
plt.close()
print("Saved: a7_evaporation.png")

# plot precipitation vs evaporation
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, tp_monthly.values, label='Precipitation', color='steelblue', linewidth=1.2)
ax.plot(time, e_monthly.values, label='Evaporation', color='orangered', linewidth=1.2)
ax.set_xlabel('Time')
ax.set_ylabel('mm/month')
ax.set_title('Monthly Precipitation vs Evaporation over Saudi Arabia (2000-2020)')
ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_precip_vs_evap.png', dpi=200)
plt.close()
print("Saved: a7_precip_vs_evap.png")

# plot runoff
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(time_mpl, ro_monthly.values, width=25, color='seagreen', alpha=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('Runoff (mm/month)')
ax.set_title('Monthly Runoff over Saudi Arabia (2000-2020)')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_runoff.png', dpi=200)
plt.close()
print("Saved: a7_runoff.png")

# plot precipitation vs runoff
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, tp_monthly.values, label='Precipitation', color='steelblue', linewidth=1.2)
ax.plot(time, ro_monthly.values, label='Runoff', color='seagreen', linewidth=1.2)
ax.set_xlabel('Time')
ax.set_ylabel('mm/month')
ax.set_title('Monthly Precipitation vs Runoff over Saudi Arabia (2000-2020)')
ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_precip_vs_runoff.png', dpi=200)
plt.close()
print("Saved: a7_precip_vs_runoff.png")

# plot water balance P minus E plus R
water_balance = tp_monthly.values - (e_monthly.values + ro_monthly.values)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['steelblue' if v >= 0 else 'crimson' for v in water_balance]
ax.bar(time_mpl, water_balance, width=25, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('P - (E + R) (mm/month)')
ax.set_title('Monthly Water Balance: P - (E + R) over Saudi Arabia (2000-2020)')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_water_balance.png', dpi=200)
plt.close()
print("Saved: a7_water_balance.png")

# annual water balance and trend
tp_annual = tp_monthly.groupby(f'{time_dim}.year').sum()
e_annual = e_monthly.groupby(f'{time_dim}.year').sum()
ro_annual = ro_monthly.groupby(f'{time_dim}.year').sum()
wb_annual = tp_annual - (e_annual + ro_annual)

years_arr = wb_annual.year.values

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# top panel shows annual P E R
ax = axes[0]
ax.bar(years_arr - 0.2, tp_annual.values, width=0.35, label='Precipitation', color='steelblue')
ax.bar(years_arr + 0.2, e_annual.values, width=0.35, label='Evaporation', color='orangered')
ax.bar(years_arr + 0.2, ro_annual.values, width=0.35, bottom=e_annual.values,
       label='Runoff', color='seagreen')
ax.set_xlabel('Year')
ax.set_ylabel('mm/year')
ax.set_title('Annual Precipitation, Evaporation, and Runoff over Saudi Arabia (2000-2020)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# bottom panel shows annual water balance with linear trend
ax = axes[1]
colors_annual = ['steelblue' if v >= 0 else 'crimson' for v in wb_annual.values]
ax.bar(years_arr, wb_annual.values, color=colors_annual, alpha=0.8)
z = np.polyfit(years_arr, wb_annual.values, 1)
p = np.poly1d(z)
ax.plot(years_arr, p(years_arr), 'k--', linewidth=2, label=f'Trend: {z[0]:.2f} mm/year/year')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('P - (E + R) (mm/year)')
ax.set_title('Annual Water Balance Trend: P - (E + R) over Saudi Arabia (2000-2020)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/15837/geo_env-/a7_annual_water_balance.png', dpi=200)
plt.close()
print("Saved: a7_annual_water_balance.png")

# print summary
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (2000-2020)")
print("=" * 60)
print(f"Mean Annual Precipitation:  {float(tp_annual.mean()):.1f} mm/year")
print(f"Mean Annual Evaporation:    {float(e_annual.mean()):.1f} mm/year")
print(f"Mean Annual Runoff:         {float(ro_annual.mean()):.1f} mm/year")
print(f"Mean Annual Water Balance:  {float(wb_annual.mean()):.1f} mm/year")
print(f"Water Balance Trend:        {z[0]:.2f} mm/year/year")
print(f"\nMonths where P > E+R:  {np.sum(water_balance > 0)} / {len(water_balance)}")
print(f"Months where P < E+R:  {np.sum(water_balance < 0)} / {len(water_balance)}")

from scipy import stats
corr, pval = stats.pearsonr(tp_monthly.values, ro_monthly.values)
print(f"\nCorrelation (Precip vs Runoff): r = {corr:.3f}, p = {pval:.2e}")

print("\nAll plots saved successfully!")
