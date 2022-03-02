from argparse import ArgumentParser

import matplotlib.pyplot as plt
import xarray as xr
import zarr


def parse_args():
    usage = "usage: python %(prog)s config_file"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "zarr_output",
        type=str,
        action="store",
        help="which zarr output file to use",
    )
    return parser.parse_args()


args = parse_args()
ds = xr.open_zarr(store=zarr.DirectoryStore(path=args.zarr_output), consolidated=False)


level = -1
varnames = ["pt"]
for varname in varnames:
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    for i in range(6):
        data = ds[varname].isel(time=0, tile=i, z=level).values
        im = ax[i].pcolormesh(data)
        ax[i].set_title(f"Tile {i}")
        plt.colorbar(im, ax=ax[i])
    fig.suptitle(f"Lowest level {varname} at initialization")
    plt.tight_layout()
    plt.savefig(f"baroclinic_initialization_{varname}.png", dpi=150)
    plt.close()
