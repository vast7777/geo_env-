import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = r'C:\Users\15837\geo_env-\GridSat_Data'


# decode_cf=False避免xarray自动解码，scale by hand）
dset = xr.open_dataset(os.path.join(data_dir, 'GRIDSAT-B1.2009.11.25.06.v02r01.nc'), decode_cf=False)
print(dset)

# 2.2 irwin_cdr
IR = np.array(dset.variables['irwin_cdr']).squeeze()
print("IR.shape:", IR.shape)

IR = np.flipud(IR)

# 2.4
IR = IR * 0.01 + 200
# 2.5
IR = IR - 273.15

# 2.6 
plt.figure(1, figsize=(12, 6))
plt.imshow(IR, extent=[-180.035, 180.035, -70.035, 70.035], aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')

# 2.7 
jeddah_lat = 21.5
jeddah_lon = 39.2
#jeddah_lat_idx = np.argmin(np.abs(dset['lat'].values - jeddah_lat))
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label='Jeddah')
plt.legend()
plt.title('GridSat-B1 Brightness Temperature 2009-11-25 06:00 UTC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('gridsat_global.png', dpi=300)
plt.show()

# 2.12 lowest
files = [
    'GRIDSAT-B1.2009.11.25.00.v02r01.nc',
    'GRIDSAT-B1.2009.11.25.03.v02r01.nc',
    'GRIDSAT-B1.2009.11.25.06.v02r01.nc',
    'GRIDSAT-B1.2009.11.25.09.v02r01.nc',
    'GRIDSAT-B1.2009.11.25.12.v02r01.nc',
]
hours = ['00:00', '03:00', '06:00', '09:00', '12:00']

# already K, no need to convert
ds_auto = xr.open_dataset(os.path.join(data_dir, files[0]))
lat = ds_auto['lat'].values
lon = ds_auto['lon'].values

lat_mask = (lat >= 19) & (lat <= 24)
lon_mask = (lon >= 36) & (lon <= 43)
jeddah_lat_idx = np.argmin(np.abs(lat - jeddah_lat))
jeddah_lon_idx = np.argmin(np.abs(lon - jeddah_lon))

print("\n2.12 各时刻吉达亮温:")
all_IR_K = []
for i, f in enumerate(files):
    ds = xr.open_dataset(os.path.join(data_dir, f))
    ir = np.array(ds['irwin_cdr']).squeeze()  # 已经是K
    all_IR_K.append(ir)
    bt_jeddah = ir[jeddah_lat_idx, jeddah_lon_idx]
    region = ir[np.ix_(lat_mask, lon_mask)]
    print(f"  {hours[i]} UTC: Jeddah BT = {bt_jeddah - 273.15:.1f} C ({bt_jeddah:.1f} K)")


# 3.2 AutoEstimator R = A * exp(-b * T^c)
A = 1.1183e11
b = 3.6382e-2
c = 1.2

cumulative_rain = np.zeros_like(all_IR_K[0])
for i, ir_k in enumerate(all_IR_K):
    R = A * np.exp(-b * ir_k**c)
    R = np.where(np.isnan(R), 0, R)
    R = np.where(R < 0, 0, R)
    cumulative_rain += R * 3.0  # 每个时刻间隔3小时

# 3.3 Cumulative rainfall 
lat_plot = (lat >= 18) & (lat <= 25)
lon_plot = (lon >= 35) & (lon <= 45)

rain_jeddah = cumulative_rain[np.ix_(lat_plot, lon_plot)]
lat_sub = lat[lat_plot]
lon_sub = lon[lon_plot]

plt.figure(figsize=(10, 8))
plt.imshow(rain_jeddah,
           extent=[lon_sub.min(), lon_sub.max(), lat_sub.min(), lat_sub.max()],
           aspect='auto', cmap='Blues')
cbar = plt.colorbar()
cbar.set_label('Cumulative rainfall (mm)')
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label='Jeddah')
plt.legend()
plt.title('Cumulative Rainfall 00:00-12:00 UTC, Nov 25, 2009')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('jeddah_rainfall_cumulative.png', dpi=300)
plt.show()

# rainfall rate at each time step
print("\n3.3 rainfall rate:")
for i, ir_k in enumerate(all_IR_K):
    T_k = ir_k[jeddah_lat_idx, jeddah_lon_idx]
    R = A * np.exp(-b * T_k**c)
    local_h = int(hours[i][:2]) + 3
    print(f"  {hours[i]} UTC (local {local_h:02d}:00): {R:.2f} mm/h, BT={T_k-273.15:.1f} C")
