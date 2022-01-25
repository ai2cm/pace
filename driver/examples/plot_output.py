import xarray as xr
import zarr


try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(
        "matplotlib is not installed, install it first with "
        "`pip install matplotlib` or similar"
    )
    raise

ds = xr.open_zarr(store=zarr.DirectoryStore(path="output.zarr"), consolidated=False)


fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()
level = -1
varname = "delp"
for i in range(6):
    temperature_anomaly = (
        ds[varname].isel(time=-1, tile=i, z=level).values
        - ds[varname].isel(time=1, tile=i, z=level).values
    )
    im = ax[i].pcolormesh(temperature_anomaly)
    ax[i].set_title(f"Tile {i}")
    plt.colorbar(im, ax=ax[i])
fig.suptitle("Lowest level temperature evolution (end - step 1)")
plt.tight_layout()
plt.show()
