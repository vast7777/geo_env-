import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools
import os


data_path = r'C:\Users\15837\geo_env-\download.nc'
dset = xarray.open_dataset(data_path)
print(dset)
t2m = np.array(dset.variables['t2m'])
tp = np.array(dset.variables['tp'])
time_dt = np.array(dset.variables['valid_time'])
latitude = np.array(dset.variables['latitude'])
longitude = np.array(dset.variables['longitude'])

# K-C and m-mm
t2m = t2m - 273.15
tp = tp * 1000


# DataFrame
df_era5 = pd.DataFrame(index=time_dt)
if t2m.ndim == 1:
    df_era5['t2m'] = t2m
    df_era5['tp'] = tp
elif t2m.ndim == 3:
    df_era5['t2m'] = t2m[:, 3, 2]
    df_era5['tp'] = tp[:, 3, 2]

# 2.7 time series plot
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_era5.index, df_era5['t2m'], color='tab:red', label='2m Temperature', linewidth=0.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)')
ax2 = ax1.twinx()
ax2.plot(df_era5.index, df_era5['tp'], label='Precipitation', linewidth=0.5)
ax2.set_ylabel('Precipitation (mm/h)', )
plt.title('ERA5 Temperature and Precipitation Time Series')
plt.tight_layout()
plt.savefig('era5_timeseries.png', dpi=150)
plt.show()

# 2.8 average annual precipitation
annual_precip = df_era5['tp'].resample('YE').mean() * 24 * 365.25
mean_annual_precip = np.nanmean(annual_precip)
print(f'\n2.8a: Annual Precipitation')
for year, val in annual_precip.items():
    print(f'  {year.year}: {val:.2f} mm/y')
print(f'  Average Annual Precipitation:{mean_annual_precip:.2f} mm/y')

# 3.1 min max and mean daily temperature
tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 21.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear

# 3.2 PE
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

# 3.3 PE time series plot
ts_index = df_era5['t2m'].resample('D').mean().index
plt.figure(figsize=(12, 5))
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential evaporation (mm/d)')
plt.title('Potential Evaporationnear')
plt.legend()
plt.tight_layout()
plt.savefig('pe_timeseries.png', dpi=150)
plt.show()

# 3.4 average annual PE (mm/y)
pe_series = pd.Series(pe, index=ts_index)
annual_pe = pe_series.resample('YE').sum()
mean_annual_pe = np.nanmean(annual_pe)
print(f'\n3.4 Average Annual PE:')
for year, val in annual_pe.items():
    print(f'  {year.year}: {val:.2f} mm/y')
print(f'  Average Annual PE: {mean_annual_pe:.2f} mm/y')

# 3.5 Reservoir evaporation loss
reservoir_area_km2 = 1.6  # km^2
reservoir_area_m2 = reservoir_area_km2 * 1e6  # m^2
evap_depth_m = mean_annual_pe / 1000  # mm -> m
volume_m3 = evap_depth_m * reservoir_area_m2
print(f'\n3.5 Reservoir evaporation loss :')
print(f'  水库面积: {reservoir_area_km2} km^2')
print(f'  年蒸发深度: {mean_annual_pe:.2f} mm = {evap_depth_m:.4f} m')
print(f'  年蒸发体积: {volume_m3:.0f} m^3 = {volume_m3/1e6:.4f} million m^3')
