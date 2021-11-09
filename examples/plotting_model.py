import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube
import time


def gather_fortran_data_column_sum(path, total_ranks, cn, var):
    """Gather Fortran diagnostics output
    Assuming the fileout has this format: atmos_custom_fine_inst.tile%RANK.nc
    where %RANK is the tile number starting from 1
    Args:
        path: direcotry to Fortran output files
        total_ranks: total number of ranks (e.g., 6, 54)
        cn: resolution (e.g., 12, 48)
        var: variable name to be extracted
    """
    start = time.time()
    ts_size = len(xr.open_dataset(path + "/atmos_custom_fine_inst.tile1.nc")["time"])
    fortran_sum = np.zeros((ts_size, total_ranks, cn, cn))
    lat = np.zeros((6, cn + 1, cn + 1))
    lon = np.zeros((6, cn + 1, cn + 1))
    for rank in range(total_ranks):
        f = xr.open_dataset(
            path + "/atmos_custom_fine_inst.tile" + str(rank + 1) + ".nc"
        )
        fortran_sum[:, rank, :, :] = f[var][:, :, :, :].sum(axis=1)
        lat[rank, :, :] = f["latb"]
        lon[rank, :, :] = f["lonb"]
        f.close()
    print(f"Load fortran {time.time() - start}s")
    return lat, lon, fortran_sum


def gather_python_data_column_sum(path, total_ranks, cn, var, ts_list):
    """Gather Python output
    Assuming the fileout has this format: pace_output_t_%TS_rank_%RANK.npy
    where %TS is the number of timesteps since the beginning, %RANK is the tile number starting from 0
    Args:
        path: direcotry to Fortran output files
        total_ranks: total number of ranks (e.g., 6, 54)
        cn: resolution (e.g., 12, 48)
        var: variable name to be extracted
        ts_list: list of timesteps to be read
    """
    start = time.time()
    var_sum = np.zeros((len(ts_list), total_ranks, cn, cn))
    t_index = 0
    for ts in ts_list:
        for rank in range(total_ranks):
            with np.load(
                path + "/pace_output_t_" + str(ts) + "_rank_" + str(rank) + ".npz",
                allow_pickle=True,
            ) as data:
                var_sum[t_index, rank, :, :] = (
                    data["arr_0"].tolist()[var][3:-4, 3:-4, :].sum(axis=-1).T
                )
        t_index += 1
    print(f"Load pace {time.time() - start}s")
    return var_sum


def load_data_column_sum(
    fortran_data_path, fortran_varname, pace_data_path, pace_varname, timesteps
):
    pace_sum = gather_python_data_column_sum(
        pace_data_path, 6, 128, pace_varname, timesteps
    )
    # Fortran data are stored here: gs://vcm-fv3gfs-data/time_series_data/c128_baroclinic_400steps
    lat, lon, fortran_sum = gather_fortran_data_column_sum(
        fortran_data_path, 6, 128, fortran_varname
    )

    return lat, lon, fortran_sum, pace_sum


if __name__ == "__main__":

    start = time.time()

    # Configuration for plots.
    # "minmax": will be scaled to the min/max value of the field
    config = {
        "qrain": {"fortran_name": "rainwat", "vmin": 0, "vmax": 0.004},
        "ua": {"fortran_name": "ucomp", "vmin": "minmax", "vmax": "minmax"},
    }

    # Chose experiments and timesteps
    experiment = "ua"
    timesteps = np.arange(5, 171, 5)

    # Data path  to load
    fortran_path = "/path/to/c128_baroclinic_400steps/fortran/"
    pace_path = "/path/to/c128_baroclinic_400steps/pace/"

    # Loading data
    lat, lon, fortran_sum, pace_sum = load_data_column_sum(
        fortran_path,
        config[experiment]["fortran_name"],
        pace_path,
        experiment,
        timesteps,
    )

    # Finding scale bounds for the plots
    if config[experiment]["vmin"] == "minmax":
        vmin = min(fortran_sum[:, :, :, :].min(), pace_sum[:, :, :, :].min())
    else:
        vmin = config[experiment]["vmin"]
    if config[experiment]["vmax"] == "minmax":
        vmax = max(fortran_sum[:, :, :, :].max(), pace_sum[:, :, :, :].max())
    else:
        vmax = config[experiment]["vmax"]

    # For each timesteps - about a fortran vs pace plot
    for ts in timesteps:

        fig, ax = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Robinson()})
        fortran_index = int(ts / 5)

        # Fortran cube sphere plot
        h = pcolormesh_cube(
            lat,
            lon,
            fortran_sum[fortran_index, :, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=plt.cm.viridis,
            ax=ax[0],
            edgecolor=None,
            linewidth=0.01,
        )
        ax[0].set_title(
            "Fortran " + config[experiment]["fortran_name"] + " ts=" + str(ts + 1)
        )
        plt.colorbar(h, ax=ax[0], label="", orientation="horizontal")

        # Pace cube sphere plot
        pace_index = int(ts / 5) - 1
        h = pcolormesh_cube(
            lat,
            lon,
            pace_sum[pace_index, :, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=plt.cm.viridis,
            ax=ax[1],
            edgecolor=None,
            linewidth=0.01,
        )
        ax[1].set_title("Python " + experiment + " ts=" + str(ts + 1))
        plt.colorbar(h, ax=ax[1], label="", orientation="horizontal")

        fig.set_size_inches([10, 4])

        # Save the dual plot
        plt.savefig(f"./plot_output/model_{ts:04d}.png")
        plt.close(fig)

    print(f"Done {time.time() - start}s")
