import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray
from collections import namedtuple
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings('ignore')

base_dir = r"C:\Users\15837\geo_env-\assignment9data"
shp_path = r"C:\Users\15837\geo_env-\assignment7\saudiSHP\Saudi_Shape.shp"
out_dir = r"C:\Users\15837\geo_env-"

# file paths for combined datasets
files = {
    'ssp126': {
        'tas': os.path.join(base_dir, 'SSP126', 'SSP126', 'Temp_126', 'Temp126.nc'),
        'pr':  os.path.join(base_dir, 'SSP126', 'SSP126', 'Precipitation_126', 'PR_126.nc'),
        'hurs': os.path.join(base_dir, 'SSP126', 'SSP126', 'Humidity_126', 'RH126.nc'),
    },
    'ssp370': {
        'tas': os.path.join(base_dir, 'SSP370', 'SSP370', 'Temp_370', 'Temp_370.nc'),
        'pr':  os.path.join(base_dir, 'SSP370', 'SSP370', 'Precipitation_370', 'pr370.nc'),
        'hurs': os.path.join(base_dir, 'SSP370', 'SSP370', 'Humudity_370', 'RH370.nc'),
    },
}

# load Saudi shapefile
gdf = gpd.read_file(shp_path)
print("Saudi shapefile loaded")

# clip a dataset to Saudi borders using shapefile
def clip_to_saudi(ds):
    ds = ds.rio.write_crs("EPSG:4326")
    if 'lat' in ds.coords:
        ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped

# Mann-Kendall test with autocorrelation correction (Hamed and Rao 1998)
def hamed_rao_mk_test(x, alpha=0.05):
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    var_s = n*(n-1)*(2*n+5)/18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t*(t-1)*(2*t+5)/18

    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, n//4)]
        n_eff = n / (1 + 2 * sum((n-i)/n * acf[i] for i in range(1, len(acf))))
        var_s *= n_eff / n

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha/2)

    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)

# Sen's slope estimator
def sens_slope(x, y):
    slopes = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)

# process a scenario: load each variable, clip to Saudi, compute annual/daily means
def process_scenario(scenario):
    print(f"\nProcessing {scenario}...")
    ds_tas = xr.open_dataset(files[scenario]['tas'])
    ds_pr = xr.open_dataset(files[scenario]['pr'])
    ds_hurs = xr.open_dataset(files[scenario]['hurs'])

    # clip to Saudi
    ds_tas = clip_to_saudi(ds_tas)
    ds_pr = clip_to_saudi(ds_pr)
    ds_hurs = clip_to_saudi(ds_hurs)

    tas = ds_tas['tas'] - 273.15  # K to C
    pr = ds_pr['pr'] * 86400  # kg/m2/s to mm/day
    hurs = ds_hurs['hurs']  # already in %

    return tas, pr, hurs

# Stull 2011 wet bulb temperature formula
def wet_bulb(T_c, RH):
    Tw = (T_c * np.arctan(0.151977 * (RH + 8.313659)**0.5)
          + np.arctan(T_c + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * (RH**1.5) * np.arctan(0.023101 * RH)
          - 4.686035)
    return Tw

# load both scenarios
tas_126, pr_126, hurs_126 = process_scenario('ssp126')
tas_370, pr_370, hurs_370 = process_scenario('ssp370')

# Part 2: annual temperature and precipitation trends
print("\n--- Part 2: Trend Analysis ---")

# annual Saudi-wide average temperature
annual_tas_126 = tas_126.mean(dim=['lat', 'lon']).resample(time='YE').mean()
annual_tas_370 = tas_370.mean(dim=['lat', 'lon']).resample(time='YE').mean()
annual_pr_126 = pr_126.mean(dim=['lat', 'lon']).resample(time='YE').sum()
annual_pr_370 = pr_370.mean(dim=['lat', 'lon']).resample(time='YE').sum()

years = annual_tas_126.time.dt.year.values

# MK test and Sen's slope
def trend_report(values, years, label):
    mk = hamed_rao_mk_test(values)
    slope = sens_slope(years, values)
    print(f"{label}:")
    print(f"  MK trend: {mk.trend}, p={mk.p:.4f}, significant: {mk.h}")
    print(f"  Sen's slope: {slope:.4f} per year")
    return mk, slope

mk_t126, slope_t126 = trend_report(annual_tas_126.values, years, "Temperature SSP1-RCP2.6")
mk_t370, slope_t370 = trend_report(annual_tas_370.values, years, "Temperature SSP3-RCP7.0")
mk_p126, slope_p126 = trend_report(annual_pr_126.values, years, "Precipitation SSP1-RCP2.6")
mk_p370, slope_p370 = trend_report(annual_pr_370.values, years, "Precipitation SSP3-RCP7.0")

# plot annual temperature
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, annual_tas_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, annual_tas_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Average Annual Temperature (C)')
ax.set_title('Average Annual Temperature over Saudi Arabia (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_annual_temp.png'), dpi=200)
plt.close()
print("Saved: a9_annual_temp.png")

# plot annual precipitation
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, annual_pr_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, annual_pr_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Annual Precipitation (mm)')
ax.set_title('Annual Precipitation over Saudi Arabia (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_annual_precip.png'), dpi=200)
plt.close()
print("Saved: a9_annual_precip.png")

# Part 3: extremes - yearly max of daily Saudi-average values
print("\n--- Part 3: Extremes ---")

# daily Saudi average, then yearly max
daily_tas_126 = tas_126.mean(dim=['lat', 'lon'])
daily_tas_370 = tas_370.mean(dim=['lat', 'lon'])
daily_pr_126 = pr_126.mean(dim=['lat', 'lon'])
daily_pr_370 = pr_370.mean(dim=['lat', 'lon'])

yearly_max_tas_126 = daily_tas_126.resample(time='YE').max()
yearly_max_tas_370 = daily_tas_370.resample(time='YE').max()
yearly_max_pr_126 = daily_pr_126.resample(time='YE').max()
yearly_max_pr_370 = daily_pr_370.resample(time='YE').max()

# plot max temperature
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, yearly_max_tas_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, yearly_max_tas_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Yearly Maximum Daily Temperature (C)')
ax.set_title('Yearly Maximum of Daily Saudi-Average Temperature (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_max_temp.png'), dpi=200)
plt.close()
print("Saved: a9_max_temp.png")

# plot max precipitation
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, yearly_max_pr_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, yearly_max_pr_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Yearly Maximum Daily Precipitation (mm)')
ax.set_title('Yearly Maximum of Daily Saudi-Average Precipitation (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_max_precip.png'), dpi=200)
plt.close()
print("Saved: a9_max_precip.png")

mk_tmax126, slope_tmax126 = trend_report(yearly_max_tas_126.values, years, "Max Temp SSP1-RCP2.6")
mk_tmax370, slope_tmax370 = trend_report(yearly_max_tas_370.values, years, "Max Temp SSP3-RCP7.0")
mk_pmax126, slope_pmax126 = trend_report(yearly_max_pr_126.values, years, "Max Precip SSP1-RCP2.6")
mk_pmax370, slope_pmax370 = trend_report(yearly_max_pr_370.values, years, "Max Precip SSP3-RCP7.0")

# Part 4: Wet bulb temperature
print("\n--- Part 4: Wet Bulb Temperature ---")

wbt_126 = wet_bulb(tas_126, hurs_126)
wbt_370 = wet_bulb(tas_370, hurs_370)

# save wbt to nc files
wbt_126.to_netcdf(os.path.join(out_dir, 'wb_126.nc'))
wbt_370.to_netcdf(os.path.join(out_dir, 'wb_370.nc'))
print("Saved: wb_126.nc, wb_370.nc")

# annual Saudi-wide average wet bulb temperature
annual_wbt_126 = wbt_126.mean(dim=['lat', 'lon']).resample(time='YE').mean()
annual_wbt_370 = wbt_370.mean(dim=['lat', 'lon']).resample(time='YE').mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, annual_wbt_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, annual_wbt_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Average Annual Wet Bulb Temperature (C)')
ax.set_title('Average Annual Wet Bulb Temperature over Saudi Arabia (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_annual_wbt.png'), dpi=200)
plt.close()
print("Saved: a9_annual_wbt.png")

# Part 5: wet bulb trend and extremes
print("\n--- Part 5: Wet Bulb Trend and Extremes ---")

mk_wbt126, slope_wbt126 = trend_report(annual_wbt_126.values, years, "Wet Bulb SSP1-RCP2.6")
mk_wbt370, slope_wbt370 = trend_report(annual_wbt_370.values, years, "Wet Bulb SSP3-RCP7.0")

# yearly max of daily Saudi-average wet bulb
daily_wbt_126 = wbt_126.mean(dim=['lat', 'lon'])
daily_wbt_370 = wbt_370.mean(dim=['lat', 'lon'])
yearly_max_wbt_126 = daily_wbt_126.resample(time='YE').max()
yearly_max_wbt_370 = daily_wbt_370.resample(time='YE').max()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, yearly_max_wbt_126.values, label='SSP1-RCP2.6', color='steelblue', linewidth=1.5)
ax.plot(years, yearly_max_wbt_370.values, label='SSP3-RCP7.0', color='orangered', linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Yearly Maximum Daily Wet Bulb Temperature (C)')
ax.set_title('Yearly Maximum of Daily Saudi-Average Wet Bulb Temperature (2015-2100)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'a9_max_wbt.png'), dpi=200)
plt.close()
print("Saved: a9_max_wbt.png")

mk_wbmax126, slope_wbmax126 = trend_report(yearly_max_wbt_126.values, years, "Max WBT SSP1-RCP2.6")
mk_wbmax370, slope_wbmax370 = trend_report(yearly_max_wbt_370.values, years, "Max WBT SSP3-RCP7.0")

# summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Annual Temp SSP1-RCP2.6: {slope_t126:.4f} C/year, trend={mk_t126.trend}, sig={mk_t126.h}")
print(f"Annual Temp SSP3-RCP7.0: {slope_t370:.4f} C/year, trend={mk_t370.trend}, sig={mk_t370.h}")
print(f"Annual Precip SSP1-RCP2.6: {slope_p126:.4f} mm/year, trend={mk_p126.trend}, sig={mk_p126.h}")
print(f"Annual Precip SSP3-RCP7.0: {slope_p370:.4f} mm/year, trend={mk_p370.trend}, sig={mk_p370.h}")
print(f"Annual WBT SSP1-RCP2.6: {slope_wbt126:.4f} C/year, trend={mk_wbt126.trend}, sig={mk_wbt126.h}")
print(f"Annual WBT SSP3-RCP7.0: {slope_wbt370:.4f} C/year, trend={mk_wbt370.trend}, sig={mk_wbt370.h}")

print("\nAll done!")
