import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube


ds = xr.open_zarr(store=zarr.DirectoryStore(path="output.zarr"), consolidated=False)
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
lat = ds["lat"].values * 180.0 / np.pi
lon = ds["lon"].values * 180.0 / np.pi
h = pcolormesh_cube(
    lat,
    lon,
    ds["ua"].isel(time=5, z=78).values,
    cmap=plt.cm.viridis,
    ax=ax,
)
fig.colorbar(h, ax=ax, location="bottom", label="u [m/s]")
plt.tight_layout()
plt.savefig("test.png")
