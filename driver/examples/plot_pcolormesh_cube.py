from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube


def parse_args():
    parser = ArgumentParser()
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
        help="variable z-level to be plotted",
    )
    parser.add_argument(
        "--zarr_output",
        type=str,
        action="store",
        help="when plotting locally, specify pace zarr output path",
        default="/model_output/output.zarr",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        action="store",
        help="minimum value in colorbar",
        default=None,
    )
    parser.add_argument(
        "--vmax",
        type=float,
        action="store",
        help="maximal value in colorbar",
        default=None,
    )
    parser.add_argument(
        "--size",
        type=int,
        action="store",
        help="npx/npy",
        default=192,
    )
    parser.add_argument(
        "--force_symmetric_colorbar",
        action="store_true",
        help="force colorbar to be symmetric around zero",
    )
    parser.add_argument(
        "--diff_init",
        action="store_true",
        help="plot python difference from first time step",
    )
    parser.add_argument(
        "--fortran_data_path",
        type=str,
        action="store",
        help="path to fortran data output if plotting difference from fortran",
        default=None,
    )
    parser.add_argument(
        "--fortran_from_wrapper",
        action="store_true",
        help="fortran data is from fv3gfs-wrapper",
    )
    parser.add_argument(
        "--fortran_var",
        type=str,
        action="store",
        help="fortran variable name",
        default=None,
    )
    parser.add_argument(
        "--diff_python_path",
        type=str,
        action="store",
        help="python reference path",
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        action="store",
        help="starting time step",
        default=0,
    )
    parser.add_argument(
        "--stop",
        type=int,
        action="store",
        help="ending time step",
        default=1,
    )
    parser.add_argument(
        "--var2D",
        action="store_true",
        help="whether variable is 2D, for diagnostics",
    )
    return parser.parse_args()


def gather_fortran_data_at_klevel(path: str, cn: int, var: str, klevel: int):
    """Gather Fortran diagnostics output
    Assuming the fileout has this format: atmos_custom_fine_inst.tile%RANK.nc
    where %RANK is the tile number starting from 1
    Args:
        path: direcotry to Fortran output files
        cn: resolution (e.g., 12, 48)
        var: variable name to be extracted
        klevel: index number in the k-axis to be read
    """
    ts_size = len(
        xr.open_dataset(path + "/atmos_custom_fine_inst.tile1.nc", decode_times=False)[
            "time"
        ]
    )
    total_tiles = 6
    fortran_data = np.zeros((ts_size, total_tiles, cn, cn))
    for rank in range(total_tiles):
        with xr.open_dataset(
            path + "/atmos_custom_fine_inst.tile" + str(rank + 1) + ".nc",
            decode_times=False,
        ) as f:
            for t in range(ts_size):
                fortran_data[t, rank, :, :] = f[var][t, klevel, :, :].T
    return fortran_data


def gather_fortran_wrapper_at_klevel(
    path: str, cn: int, var: str, klevel: int, ts_size: int
):
    total_tiles = 6
    fortran_data = np.zeros((ts_size, total_tiles, cn, cn))
    for rank in range(total_tiles):
        for t in range(ts_size):
            with xr.open_dataset(
                path + f"/outstate_{t}_{rank}.nc",
            ) as f:
                fortran_data[t, rank, :, :] = f[var][klevel, :, :].T
    return fortran_data


if __name__ == "__main__":
    args = parse_args()
    if (
        sum(x is not None for x in [args.fortran_data_path, args.diff_python_path])
        + args.diff_init
    ) > 1:
        raise RuntimeError(
            "Scirpt called with confilicting options between: \
            Diff init, diff python and diff to fortran"
        )
    if args.fortran_data_path is not None:
        if args.fortran_var is None:
            raise ValueError(
                "You must specify the variable name (fortran_var) \
                    to be subtracted in Fortran data."
            )
        if args.fortran_from_wrapper:
            fortran = gather_fortran_wrapper_at_klevel(
                args.fortran_data_path, args.size, args.fortran_var, args.zlevel, 20
            )
        else:
            fortran = gather_fortran_data_at_klevel(
                args.fortran_data_path, args.size, args.fortran_var, args.zlevel
            )
    if args.fortran_var is not None and args.fortran_data_path is None:
        raise ValueError(
            "You must specify the path (fortran_data_path) to Fortran data."
        )
    if args.diff_python_path is not None:
        ds_ref = xr.open_zarr(
            store=zarr.DirectoryStore(path=args.diff_python_path), consolidated=False
        )
        if args.var2D:
            python_ref = ds_ref[args.variable][:, :, 0 : args.size, 0 : args.size]
        else:
            python_ref = ds_ref[args.variable][:, :, 0 : args.size, 0 : args.size].isel(
                z=args.zlevel
            )

    ds = xr.open_zarr(
        store=zarr.DirectoryStore(path=args.zarr_output), consolidated=False
    )
    python_lat = ds["lat"].values * 180.0 / np.pi
    python_lon = ds["lon"].values * 180.0 / np.pi
    if args.diff_init:
        if args.fortran_data_path is not None:
            raise ValueError(
                "You cannot plot the difference from Fortran \
                    when plotting the python difference from the first time step."
            )
        if args.var2D:
            python_init = (
                ds[args.variable][:, :, 0 : args.size, 0 : args.size]
                .isel(time=0)
                .values
            )
        else:
            python_init = (
                ds[args.variable][:, :, 0 : args.size, 0 : args.size, :]
                .isel(time=0, z=args.zlevel)
                .values
            )
    for t in range(args.start, args.stop):
        if args.var2D:
            python = (
                ds[args.variable][:, :, 0 : args.size, 0 : args.size]
                .isel(time=t)
                .values
            )
        else:
            python = (
                ds[args.variable][:, :, 0 : args.size, 0 : args.size, :]
                .isel(time=t, z=args.zlevel)
                .values
            )
        if args.fortran_data_path is not None:
            plotted_data = python - fortran[t, :]
        elif args.diff_init:
            plotted_data = python - python_init
        elif args.diff_python_path is not None:
            plotted_data = python - python_ref.isel(time=t).values
        else:
            plotted_data = python
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
        if args.force_symmetric_colorbar:
            abs_max = np.abs(plotted_data).max()
            h = pcolormesh_cube(
                python_lat,
                python_lon,
                plotted_data,
                cmap=plt.cm.bwr,
                ax=ax,
                vmin=-abs_max,
                vmax=abs_max,
            )
        elif args.vmin is not None and args.vmax is not None:
            h = pcolormesh_cube(
                python_lat,
                python_lon,
                plotted_data,
                ax=ax,
                cmap=plt.cm.bwr if args.vmin == -1 * args.vmax else plt.cm.viridis,
                vmin=args.vmin,
                vmax=args.vmax,
            )
        else:
            h = pcolormesh_cube(
                python_lat,
                python_lon,
                plotted_data,
                ax=ax,
            )
        fig.colorbar(h, ax=ax, location="bottom", label=f"{args.variable}")
        title = args.experiment.replace("_", " ")
        fig.suptitle(f"{title}: {args.variable}, z={args.zlevel}, timestep={t+1}")
        ax.annotate(
            "Generated on " + datetime.now().strftime("%m/%d/%y %H:%M:%S"),
            xy=(1.0, -0.6),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=8,
        )
        plt.tight_layout()
        # change this if not using saurs/docker
        save_path = "/work/"
        plt.savefig(
            f"{save_path}{args.experiment}_{args.variable}_time_{t:02d}.png",
            dpi=150,
        )
        plt.close()
