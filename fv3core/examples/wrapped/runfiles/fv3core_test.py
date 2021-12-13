import copy
import sys

import mpi4py
import numpy
import numpy as np
import xarray as xr
import yaml

import fv3core
import fv3core._config as spec
import fv3gfs.wrapper as wrapper
from fv3core.testing import translate
from pace.util import (
    X_DIMS,
    X_INTERFACE_DIM,
    Y_DIMS,
    Y_INTERFACE_DIM,
    Z_DIMS,
    Z_INTERFACE_DIM,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    Quantity,
    QuantityFactory,
    SubtileGridSizer,
    io,
)


sys.path.append("/usr/local/serialbox/python")
import serialbox  # noqa: E402


# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this
# sets the stack size to unlimited

# Run using mpirun -n 6 python3 fv3core_test.py


def transpose(state, dims, npz, npx, npy):
    return_state = {}
    for name, quantity in state.items():
        if name == "time":
            return_state[name] = quantity
        else:
            if len(quantity.storage.shape) == 2:
                data_3d = numpy.ascontiguousarray(
                    numpy.broadcast_to(
                        quantity.data[:, :, None],
                        (quantity.data.shape[0], quantity.data.shape[1], npz + 1),
                    )
                )
                quantity_3d = Quantity.from_data_array(
                    xr.DataArray(
                        data_3d,
                        attrs=quantity.attrs,
                        dims=[quantity.dims[0], quantity.dims[1], Z_INTERFACE_DIM],
                    ),
                    origin=(quantity.origin[0], quantity.origin[1], 0),
                    extent=(quantity.extent[0], quantity.extent[1], npz + 1),
                )
                quantity_3d.metadata.gt4py_backend = "numpy"
                return_state[name] = quantity_3d.transpose(dims)
            elif len(quantity.storage.shape) == 1:
                data_3d = numpy.tile(quantity.data, (npx + 6, npy + 6, 1))
                quantity_3d = Quantity.from_data_array(
                    xr.DataArray(
                        data_3d,
                        attrs=quantity.attrs,
                        dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, quantity.dims[0]],
                    ),
                    origin=(0, 0, quantity.origin[0]),
                    extent=(npx, npy, quantity.extent[0]),
                )
                quantity_3d.metadata.gt4py_backend = "numpy"
                return_state[name] = quantity_3d.transpose(dims)
            else:
                return_state[name] = quantity.transpose(dims)

    return return_state


def convert_3d_to_2d(state, field_levels):
    return_state = state
    for field in field_levels.keys():
        quantity = state[field]
        # Assuming we've already transposed from xyz to zyx
        data_2d = quantity.data[
            field_levels[field], :, :
        ]  # take the bottom level since they should all be the same
        quantity_2d = Quantity.from_data_array(
            xr.DataArray(
                data_2d, attrs=quantity.attrs, dims=[quantity.dims[1], quantity.dims[2]]
            ),
            origin=(quantity.origin[1], quantity.origin[2]),
            extent=(quantity.extent[1], quantity.extent[2]),
        )
        return_state[field] = quantity_2d
    return return_state


def convert_3d_to_1d(state, field_names):
    return_state = state
    for field in field_names:
        quantity = state[field]
        # Assuming we've already transposed from xyz to zyx
        data_1d = quantity.data[
            :, 0, 0
        ]  # take the first column since they should be the same
        quantity_1d = Quantity.from_data_array(
            xr.DataArray(data_1d, attrs=quantity.attrs, dims=[quantity.dims[0]]),
            origin=[quantity.origin[0]],
            extent=[quantity.extent[0]],
        )
        return_state[field] = quantity_1d
    return return_state


if __name__ == "__main__":

    # read in the namelist
    spec.set_namelist("input.nml")
    dt_atmos = spec.namelist.dt_atmos

    # set backend
    backend = "numpy"
    fv3core.set_backend(backend)

    # get another namelist for the communicator??
    nml2 = yaml.safe_load(
        open("/fv3core/examples/wrapped/config/c12_6ranks_standard.yml", "r")
    )["namelist"]

    sizer = SubtileGridSizer.from_namelist(nml2)
    allocator = QuantityFactory.from_backend(sizer, fv3core.get_backend())

    # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    cube_comm = CubedSphereCommunicator(
        comm, CubedSpherePartitioner.from_namelist(nml2)
    )

    # Set the names of quantities in State. This is everything coming from
    # wrapper.initialize.
    initial_names = [
        "specific_humidity",
        "cloud_water_mixing_ratio",
        "rain_mixing_ratio",
        "snow_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "graupel_mixing_ratio",
        "ozone_mixing_ratio",
        "cloud_fraction",
        "air_temperature",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "logarithm_of_interface_pressure",
        "x_wind",
        "y_wind",
        "vertical_wind",
        "x_wind_on_c_grid",
        "y_wind_on_c_grid",
        "total_condensate_mixing_ratio",
        "interface_pressure",
        "surface_geopotential",
        "interface_pressure_raised_to_power_of_kappa",
        "surface_pressure",
        "vertical_pressure_velocity",
        "atmosphere_hybrid_a_coordinate",
        "atmosphere_hybrid_b_coordinate",
        "accumulated_x_mass_flux",
        "accumulated_y_mass_flux",
        "accumulated_x_courant_number",
        "accumulated_y_courant_number",
        "dissipation_estimate_from_heat_source",
        "eastward_wind",
        "northward_wind",
        "layer_mean_pressure_raised_to_power_of_kappa",
        "time",
    ]

    # this contains all the names needed to run the dycore.
    all_names = copy.deepcopy(initial_names)
    all_names.append("turbulent_kinetic_energy")

    levels_of_2d_variables = {
        "surface_geopotential": -1,
        "surface_pressure": 0,
    }

    names_of_1d_variables = [
        "atmosphere_hybrid_a_coordinate",
        "atmosphere_hybrid_b_coordinate",
    ]

    # get grid from serialized data
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        "/fv3core/test_data/c12_6ranks_standard",
        "Generator_rank" + str(rank),
    )
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = translate.TranslateGrid(grid_data, rank, backend=backend).python_grid()
    fv3core._config.set_grid(grid)

    # startup
    wrapper.initialize()

    # add things to State
    origin = (0, 3, 3)
    extent = (spec.namelist.npz, spec.namelist.npy - 1, spec.namelist.npx - 1)
    arr = np.zeros(
        (spec.namelist.npz + 1, spec.namelist.npy + 6, spec.namelist.npx + 6)
    )
    turbulent_kinetic_energy = Quantity.from_data_array(
        xr.DataArray(
            arr,
            attrs={"fortran_name": "qsgs_tke", "units": "m**2/s**2"},
            dims=["z", "y", "x"],
        ),
        origin=origin,
        extent=extent,
    )
    u_tendency = Quantity.from_data_array(
        xr.DataArray(
            arr.reshape(
                (spec.namelist.npx + 6, spec.namelist.npy + 6, spec.namelist.npz + 1)
            ),
            attrs={"fortran_name": "u_dt", "units": "m/s**2"},
            dims=["x", "y", "z"],
        ),
        origin=(3, 3, 0),
        extent=(spec.namelist.npx - 1, spec.namelist.npy - 1, spec.namelist.npz),
    )
    v_tendency = Quantity.from_data_array(
        xr.DataArray(
            arr.reshape(
                (spec.namelist.npx + 6, spec.namelist.npy + 6, spec.namelist.npz + 1)
            ),
            attrs={"fortran_name": "v_dt", "units": "m/s**2"},
            dims=["x", "y", "z"],
        ),
        origin=(3, 3, 0),
        extent=(spec.namelist.npx - 1, spec.namelist.npy - 1, spec.namelist.npz),
    )

    turbulent_kinetic_energy.metadata.gt4py_backend = fv3core.get_backend()
    u_tendency.metadata.gt4py_backend = fv3core.get_backend()
    v_tendency.metadata.gt4py_backend = fv3core.get_backend()

    n_tracers = 6

    state = wrapper.get_state(allocator=allocator, names=initial_names)
    cube_comm.halo_update(state["surface_geopotential"])
    dycore = fv3core.DynamicalCore(
        comm=cube_comm,
        grid_data=spec.grid.grid_data,
        stencil_factory=spec.grid.stencil_factory,
        damping_coefficients=spec.grid.damping_coefficients,
        config=spec.namelist.dynamical_core,
        phis=state["surface_geopotential"],
    )

    fvsubgridz = fv3core.DryConvectiveAdjustment(
        spec.grid.grid_indexing,
        spec.namelist.nwat,
        spec.namelist.fv_sg_adj,
        spec.namelist.n_sponge,
        spec.namelist.hydrostatic,
    )
    # Step through time
    for i in range(wrapper.get_step_count()):
        print("STEP IS ", i)
        if i > 0:
            state = wrapper.get_state(allocator=allocator, names=initial_names)
        state["turbulent_kinetic_energy"] = turbulent_kinetic_energy
        if i == 0:
            io.write_state(state, "instate_{0}.nc".format(rank))
        state = transpose(
            state,
            [X_DIMS, Y_DIMS, Z_DIMS],
            spec.namelist.npz,
            spec.namelist.npx,
            spec.namelist.npy,
        )

        dycore.step_dynamics(
            state,
            wrapper.flags.consv_te,
            wrapper.flags.do_adiabatic_init,
            dt_atmos,
            wrapper.flags.n_split,
        )
        if spec.namelist.fv_sg_adj > 0:
            state["eastward_wind_tendency_due_to_physics"] = u_tendency
            state["northward_wind_tendency_due_to_physics"] = v_tendency
            fvsubgridz(state, dt_atmos)

        newstate = {}
        for name, quantity in state.items():
            if name not in [
                "eastward_wind_tendency_due_to_physics",
                "northward_wind_tendency_due_to_physics",
                "turbulent_kinetic_energy",
            ]:
                newstate[name] = quantity

        newstate = transpose(
            newstate,
            [Z_DIMS, Y_DIMS, X_DIMS],
            spec.namelist.npz,
            spec.namelist.npx,
            spec.namelist.npy,
        )
        newstate = convert_3d_to_2d(newstate, levels_of_2d_variables)
        newstate = convert_3d_to_1d(newstate, names_of_1d_variables)
        wrapper.set_state(newstate)
        wrapper.step_physics()
        wrapper.save_intermediate_restart_if_enabled()
    state = wrapper.get_state(allocator=allocator, names=initial_names)

    io.write_state(state, "outstate_{0}.nc".format(rank))
    wrapper.cleanup()
