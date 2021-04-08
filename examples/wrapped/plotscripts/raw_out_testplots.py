from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


usage = "usage: python %(prog)s <output directory> [optional 2nd output directory] [other options]"  # noqa: E501
parser = ArgumentParser(usage=usage)

parser.add_argument(
    "model_dir", type=str, action="store", help="directory containing outputs to plot"
)
parser.add_argument(
    "reference_dir",
    type=str,
    action="store",
    help="directory containing outputs to compare to",
    nargs="?",
)
args = parser.parse_args()

rainbow_colorsheme = [
    "#ec1b8c",
    "#a6228e",
    "#20419a",
    "#0085cc",
    "#03aeef",
    "#03aa4f",
    "#c8da2c",
    "#fff200",
    "#f99e1c",
    "#ed1c24",
]

np.set_printoptions(precision=14)

##################
# Data Wrangling #
##################

datafiles = [
    r"outstate_0.nc",
    r"outstate_1.nc",
    r"outstate_2.nc",
    r"outstate_3.nc",
    r"outstate_4.nc",
    r"outstate_5.nc",
]

surface_pressure_plots = []
surface_temperature_plots = []

relative_errors: Dict[str, np.ndarray] = {}

for filename in datafiles:
    fname_model = args.model_dir + filename
    model_data = Dataset(fname_model, "r")
    nc_attrs = model_data.ncattrs()

    surface_pressure = (
        model_data.variables["surface_pressure"][:].data / 100.0
    )  # convert to hPa
    temperature = model_data.variables["air_temperature"][:].data

    surface_temperature = temperature[-1, :, :]  # temperature at bottom

    if args.reference_dir:
        fname_ref = args.reference_dir + filename
        reference_data = Dataset(fname_ref, "r")
        surface_pressure2 = (
            reference_data.variables["surface_pressure"][:].data / 100.0
        )  # convert to hPa
        temperature2 = reference_data.variables["air_temperature"][:].data
        surface_temperature2 = temperature2[-1, :, :]  # field at 850 hPa

        surface_pressure_diff = (
            surface_pressure - surface_pressure2
        ) / surface_pressure2
        temperature_diff = (
            surface_temperature - surface_temperature2
        ) / surface_temperature2

        surface_pressure_plots.append(surface_pressure_diff)
        surface_temperature_plots.append(temperature_diff)

        # savin' variables
        for var in list(model_data.variables):
            if "time" not in var:
                if var in reference_data.variables.keys():
                    if var not in relative_errors.keys():
                        relative_errors[var] = []
                    field1 = model_data[var][:]
                    field2 = reference_data[var][:]
                    relative_errors[var].append((field1 - field2) / field2)

    else:
        surface_pressure_plots.append(surface_pressure)
        surface_temperature_plots.append(surface_temperature)


####################
# Doin' some stats #
####################
if args.reference_dir:
    pmeans = []
    tmeans = []
    vmeans = []

    p = np.array(surface_pressure_plots)
    tp = np.array(surface_temperature_plots)
    pres_date = np.concatenate(
        (p[0, :, :], p[1, :, :], p[2, :, :], p[3, :, :], p[4, :, :], p[5, :, :]), axis=0
    )
    pmeans.append(pres_date.mean())

    surface_temperature_date = np.concatenate(
        (tp[0, :, :], tp[1, :, :], tp[2, :, :], tp[3, :, :], tp[4, :, :], tp[5, :, :]),
        axis=0,
    )
    tmeans.append(surface_temperature_date.mean())

    print(pmeans)
    print(tmeans)

    for var in relative_errors.keys():
        var_tiles = np.concatenate(
            (
                relative_errors[var][0],
                relative_errors[var][1],
                relative_errors[var][2],
                relative_errors[var][3],
                relative_errors[var][4],
                relative_errors[var][5],
            ),
            axis=0,
        ).flatten()
        vmeans.append(var_tiles.mean())
        print(var, np.mean(np.abs(var_tiles.flatten())))

################
# Making Plots #
################

plotdir = "raw_state"

# Stuff for to make plots more prettier
axwidth = 3
axlength = 12
fontsize = 20
linewidth = 6
labelsize = 20

plt.rc("text.latex", preamble=r"\boldmath")
plt.rc("text", usetex=True)

# Plot params:
minlon = 1
maxlon = 48
minlat = 1
maxlat = 48

colorbar_array = [0.1, 0.05, 0.8, 0.05]
bottom_adjust = 0.15

pressure_levels = 5
temperature_levels = 5

if args.reference_dir:
    post = "diff"
else:
    post = "range"

tilestrs = [
    r"$\mathrm{Tile\ 1}$",
    r"$\mathrm{Tile\ 2}$",
    r"$\mathrm{Tile\ 3}$",
    r"$\mathrm{Tile\ 4}$",
    r"$\mathrm{Tile\ 5}$",
    r"$\mathrm{Tile\ 6}$",
]

# Pressure plots
fig1, axs1 = plt.subplots(3, 2)
levs = []
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        if k == 0:
            cs = axs1[ii, jj].contourf(
                np.log10(np.abs(surface_pressure_plots[k])), pressure_levels
            )
            levs = cs.levels
        else:
            cs = axs1[ii, jj].contourf(
                np.log10(np.abs(surface_pressure_plots[k])), levs
            )
        axs1[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig1.subplots_adjust(bottom=bottom_adjust)
cax = fig1.add_axes(colorbar_array)
cbar = fig1.colorbar(cs, cax=cax, orientation="horizontal")

fig1.suptitle(r"$\mathrm{Log_{10}\ Surface\ Pressure}$")

plt.savefig("{0}/tile_pressure0_{1}.png".format(plotdir, post))

# Surface Temperature plots
fig2, axs2 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        if k == 0:
            cs = axs2[ii, jj].contourf(
                np.log10(np.abs(surface_temperature_plots[k])), temperature_levels
            )
            levs = cs.levels
        else:
            cs = axs2[ii, jj].contourf(
                np.log10(np.abs(surface_temperature_plots[k])), levs
            )
        axs2[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig2.subplots_adjust(bottom=bottom_adjust)
cax = fig2.add_axes(colorbar_array)
cbar = fig2.colorbar(cs, cax=cax, orientation="horizontal")

fig2.suptitle(r"$\mathrm{Log_{10}\ Bottom\ Temperature}$")
plt.savefig("{0}/tile_bot_temperature0_{1}.png".format(plotdir, post))
