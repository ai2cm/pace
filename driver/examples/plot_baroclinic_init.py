from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube


def parse_args():
    usage = "usage: python %(prog)s config_file"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "zarr_output",
        type=str,
        action="store",
        help="which zarr output file to use",
    )

    parser.add_argument(
        "experiment",
        type=str,
        action="store",
        help="experiment name",
    )

    parser.add_argument(
        "variable",
        type=str,
        action="store",
        help="variable name to be plotted",
    )

    parser.add_argument(
        "zlevel",
        type=int,
        action="store",
        help="variable zlevel to be plotted",
    )

    return parser.parse_args()


args = parse_args()
ds = xr.open_zarr(store=zarr.DirectoryStore(path=args.zarr_output), consolidated=False)
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
lat = ds["lat"].values * 180.0 / np.pi
lon = ds["lon"].values * 180.0 / np.pi
h = pcolormesh_cube(
    lat,
    lon,
    ds[args.variable].isel(time=0, z=args.zlevel).values,
    cmap=plt.cm.viridis,
    ax=ax,
)
fig.colorbar(h, ax=ax, location="bottom", label=f"{args.variable}")
title = args.experiment.replace("_", " ")
fig.suptitle(f"{title}: {args.variable}, z={args.zlevel}")
ax.annotate(
    "Generated on " + datetime.now().strftime("%m/%d/%y %H:%M:%S"),
    xy=(1.0, -0.6),
    xycoords="axes fraction",
    ha="right",
    va="center",
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    f"/work/{args.experiment}_baroclinic_initialization_{args.variable}.png", dpi=150
)
