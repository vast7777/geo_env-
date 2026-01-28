import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
dset = xr.open_dataset(r"C:\Users\15837\geo_env-\SRTMGL1_NC.003_Data\N21E039.SRTMGL1_NC.nc")
pdb.set_trace()
DEM=np.array(dset.variables["SRTMGL1_DEM"])
dset.close()
pdb.set_trace()
print('DEM.shape=',DEM.shape)
plt.imshow(DEM)
cbar=plt.colorbar()
cbar.set_label('Elevation (m asl)')
plt.savefig("assignment_1.png", dpi=300)
plt.show()

