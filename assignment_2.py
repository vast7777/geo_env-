import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

file_path = r"C:\Users\15837\geo_env-\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc"

dset = xr.open_dataset(file_path)

# 1. Inspect the contents of the netCDF file using dset

print(dset)

# 2. Which variables does the netCDF contain?
print("=" * 60)
print(dset.keys())

# 3.Access the air temperature variable using dset[’tas’]. What are the dimensions of the air temperature variable?
print("=" * 60)
tas = dset['tas']
print("Dimensions:", tas.dims)
print("Shape:", tas.shape)

# 4. What kind of data is this: raster, vector, or point?  RASTER data

# 5.What is the data type of the air temperature variable: integer, single, or double? Find out using the command dset[’tas’].dtype.
print("=" * 60)
print("dtype:",tas.dtype)

# 6. Is this the optimal data type for air temperature data? Yes,less storage than float64 (double precision) and sufficient precision for air temperature data.

# 7. Temporal span of each netCDF file
print("=" * 60)
time = dset['time']
print(str(time.values[0]))
print(str(time.values[-1]))
print("span:", len(time),"months")

# 8. U What are the units of the air temperature data
print("=" * 60)
print("Units:", dset['tas'].attrs.get('units'))

# 9. What is the spatial and temporal resolution of the air temperature data
print("=" * 60)
lat = dset['lat'].values
lon = dset['lon'].values
lat_resolution = lat[1] - lat[0]
lon_resolution = lon[1] - lon[0]
print(f"Spatial resolution: {lat_resolution} deg (lat) x {lon_resolution} deg (lon)")

# 10.What is the meaning of ssp in the file names?


# 11. Type of model

# Part 3: Creation of Climate Change Map
# 1. Calculate the mean air temperature map for 1850–1900 (also known as the pre-industrial period) using the command:
dset = xr.open_dataset(
    r"C:\Users\15837\geo_env-\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc"
)

mean_1850_1900 = np.mean(dset['tas'].sel(time=slice('18500101', '19001231')), axis=0)
mean_1850_1900 = np.array(mean_1850_1900)

# Explore the properties of the variable you just created by using mean_1850_1900.shape and mean_1850_1900.dtype. What are the lat/lon coordinates for plotting? You can get them from dset[’lat’] and dset[’lon’].
print("mean_1850_1900.shape:", mean_1850_1900.shape)
print("mean_1850_1900.dtype:", mean_1850_1900.dtype)

# Get lat/lon coordinates for plotting
lat = dset['lat'].values
lon = dset['lon'].values

# Calculate mean air temperature maps for 2071–2100 for each climate scenario. 
# SSP1-1.9 
dset_ssp119 = xr.open_dataset(
    r"C:\Users\15837\geo_env-\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc"
)
mean_2071_2100_ssp119 = np.mean(dset_ssp119['tas'].sel(time=slice('20710101', '21001231')), axis=0)
mean_2071_2100_ssp119 = np.array(mean_2071_2100_ssp119)

# SSP2-4.5 
dset_ssp245 = xr.open_dataset(
    r"C:\Users\15837\geo_env-\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc"
)
mean_2071_2100_ssp245 = np.mean(dset_ssp245['tas'].sel(time=slice('20710101', '21001231')), axis=0)
mean_2071_2100_ssp245 = np.array(mean_2071_2100_ssp245)

# SSP5-8.5
dset_ssp585 = xr.open_dataset(
    r"C:\Users\15837\geo_env-\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc"
)
mean_2071_2100_ssp585 = np.mean(dset_ssp585['tas'].sel(time=slice('20710101', '21001231')), axis=0)
mean_2071_2100_ssp585 = np.array(mean_2071_2100_ssp585)

# 3. Compute temperature differences between 2071-2100 and 1850-1900
diff_ssp119 = mean_2071_2100_ssp119 - mean_1850_1900
diff_ssp245 = mean_2071_2100_ssp245 - mean_1850_1900
diff_ssp585 = mean_2071_2100_ssp585 - mean_1850_1900
###vmin and vmax for color scale 
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
vmin = 0
vmax = 8

# SSP1-1.9 plot RdYlBu_r is a diverging colormap that goes from red (high values) to yellow (medium values) to blue (low values). The _r suffix means it is reversed, so it goes from blue (high values) to yellow (medium values) to red (low values). This is often used for temperature change maps, where we want to show warming (positive changes) in red and cooling (negative changes) in blue.
im1 = axes[0].imshow(diff_ssp119, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
axes[0].set_title('Temperature Change: SSP1-1.9 (2071-2100 vs 1850-1900)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
plt.colorbar(im1, ax=axes[0], label='Temperature Change (K)')

# SSP2-4.5 plot
im2 = axes[1].imshow(diff_ssp245, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
axes[1].set_title('Temperature Change: SSP2-4.5 (2071-2100 vs 1850-1900)')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
plt.colorbar(im2, ax=axes[1], label='Temperature Change (K)')

# SSP5-8.5 plot
im3 = axes[2].imshow(diff_ssp585, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
axes[2].set_title('Temperature Change: SSP5-8.5 (2071-2100 vs 1850-1900)')
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('Latitude')
plt.colorbar(im3, ax=axes[2], label='Temperature Change (K)')

plt.savefig('climate_change_map.png', dpi=300)
plt.show()
pdb.set_trace()
