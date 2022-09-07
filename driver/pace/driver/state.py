import dataclasses
from dataclasses import fields
from typing import Union

import numpy as np
from pace.util.constants import N_HALO_DEFAULT
import xarray as xr

import pace.dsl.gt4py_utils as gt_utils
import pace.physics
import pace.util
import pace.util.grid
from pace import fv3core
from pace.dsl.gt4py_utils import is_gpu_backend
from pace.util import N_HALO_DEFAULT as halo
from pace.util.grid import DampingCoefficients


try:
    import cupy as cp
except ImportError:
    cp = None


@dataclasses.dataclass()
class TendencyState:
    """
    Accumulated tendencies from physical parameterizations to be applied
    to the dynamical core model state.
    """

    u_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    v_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    pt_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "temperature_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "K/s",
            "intent": "inout",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory: pace.util.QuantityFactory) -> "TendencyState":
        initial_quantities = {}
        for _field in dataclasses.fields(cls):
            initial_quantities[_field.name] = quantity_factory.zeros(
                _field.metadata["dims"],
                _field.metadata["units"],
                dtype=float,
            )
        return cls(**initial_quantities)


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: pace.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData

    @classmethod
    def load_state_from_restart(
        cls, restart_path: str, driver_config, fortran_data: bool = False
    ) -> "DriverState":
        comm = driver_config.comm_config.get_comm()
        communicator = pace.util.CubedSphereCommunicator.from_layout(
            comm=comm, layout=driver_config.layout
        )
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=driver_config.nx_tile,
            ny_tile=driver_config.nx_tile,
            nz=driver_config.nz,
            n_halo=pace.util.N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=driver_config.layout,
            tile_partitioner=communicator.partitioner.tile,
            tile_rank=communicator.tile.rank,
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, backend=driver_config.stencil_config.compilation_config.backend
        )
        state = _restart_driver_state(
            restart_path,
            communicator.rank,
            quantity_factory,
            communicator,
            fortran_data,
        )
        return state

    def save_state(self, comm, restart_path: str = "RESTART"):
        from pathlib import Path

        Path(restart_path).mkdir(parents=True, exist_ok=True)
        current_rank = str(comm.Get_rank())
        self.dycore_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_dycore_state_{current_rank}.nc"
        )
        self.physics_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_physics_state_{current_rank}.nc"
        )
        # we can also convert the state to Fortran's restart format using
        # code similar to this commented code. We don't need this feature right
        # now so we haven't implemented it, but this is a good starter.
        """
        xr.Dataset(
            data_vars={
                "cld_amt": state.dycore_state.qcld.data_array,
                "graupel": state.dycore_state.qgraupel.data_array,
                "ice_wat": state.dycore_state.qice.data_array,
                "liq_wat": state.dycore_state.qliquid.data_array,
                "o3mr": state.dycore_state.qo3mr.data_array,
                "rainwat": state.dycore_state.qrain.data_array,
                "sgs_tke": state.dycore_state.qsgs_tke.data_array,
                "snowwat": state.dycore_state.qsnow.data_array,
                "sphum": state.dycore_state.qvapor.data_array,
            }
        ).rename(
            {
                "z": "zaxis_1",
                "x": "xaxis_1",
                "y": "yaxis_1",
            }
        ).transpose(
            "zaxis_1", "yaxis_1", "xaxis_1"
        ).expand_dims(
            dim="Time", axis=0
        ).to_netcdf(os.path.join(path, f"fv_tracer.res.tile{rank + 1}.nc"))
        """


def _overwrite_state_from_restart(
    path: str,
    rank: int,
    state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Args:
        path: path to restart files
        rank: current rank number
        state: an empty state
        restart_file_prefix: file prefix name to read

    Returns:
        state: new state filled with restart files
    """
    df = xr.open_dataset(path + f"/{restart_file_prefix}_{rank}.nc")
    for _field in fields(type(state)):
        if "units" in _field.metadata.keys():
            if is_gpu_backend:
                if "physics" in restart_file_prefix:
                    state.__dict__[_field.name][:] = gt_utils.asarray(
                        df[_field.name].data[:], to_type=cp.ndarray
                    )
                else:
                    state.__dict__[_field.name].data[:] = gt_utils.asarray(
                        df[_field.name].data[:], to_type=cp.ndarray
                    )
            else:
                state.__dict__[_field.name].data[:] = df[_field.name].data[:]
    return state


def _overwrite_state_from_fortran_restart(
    path: str,
    communicator: pace.util.CubedSphereCommunicator,
    state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Args:
        path: path to restart files
        rank: current rank number
        state: an empty state
        restart_file_prefix: file prefix name to read

    Returns:
        state: new state filled with restart files
    """

    state_dict = _driver_state_to_dict(state, restart_file_prefix, is_gpu_backend)

    state_dict = pace.util.open_restart(path, communicator, fortran_restart=True)

    state = _dict_state_to_driver_state(
        state_dict, state, restart_file_prefix, is_gpu_backend
    )

    return state


def _driver_state_to_dict(
    driver_state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Takes a Pace driver state
    and returns a dict of state quantities with their Fortran names
    """

    dict_state = {}

    for _field in fields(type(driver_state)):
        if "units" in _field.metadata.keys():
            if is_gpu_backend:
                if "phy" in restart_file_prefix:
                    dict_state[_field.name] = driver_state.__dict__[_field.name][:]
                else:
                    dict_state[_field.name] = driver_state.__dict__[_field.name].data[:]
            else:
                dict_state[_field.name] = driver_state.__dict__[_field.name].data[:]

    return dict_state


def _dict_state_to_driver_state(
    fortran_state: dict,
    driver_state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Takes a dict of state quantities with their Fortran names and a driver state
    and populates the driver state with quantities from the dict.
    """

    for _field in fields(type(driver_state)):
        if "units" in _field.metadata.keys():
            if is_gpu_backend:
                if "phy" in restart_file_prefix:
                    driver_state.__dict__[_field.name][:] = fortran_state[_field.name]
                else:
                    driver_state.__dict__[_field.name].data[:] = fortran_state[
                        _field.name
                    ]
            else:
                if (
                    _field.name in fortran_state
                    and _field.name in driver_state.__dict__.keys()
                ):
                    if isinstance(
                        driver_state, pace.physics.physics_state.PhysicsState
                    ):
                        driver_state.__dict__[_field.name][
                            halo : -halo - 1, halo : -halo - 1, :-1
                        ] = np.transpose(fortran_state[_field.name].data)
                    else:
                        driver_state.__dict__[_field.name].view[:] = np.transpose(
                            fortran_state[_field.name].data
                        )

    return driver_state


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: pace.util.QuantityFactory,
    communicator: pace.util.CubedSphereCommunicator,
    fortran_data: bool = False,
    fortran_grid: bool = False,
):
    metric_terms = pace.util.grid.MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )

    if fortran_grid:
        ds = xr.open_dataset(path+"/grid_spec.tile%s.nc" % (rank+1))
        lon_fort = ds.grid_lon.data
        lat_fort = ds.grid_lat.data
        # metric_terms = metric_terms.from_lat_lon(lat_fort, lon_fort)

        new_grid = np.empty(metric_terms.grid.data.shape)
        new_grid[N_HALO_DEFAULT:-N_HALO_DEFAULT, N_HALO_DEFAULT:-N_HALO_DEFAULT, 0] = lon_fort
        new_grid[N_HALO_DEFAULT:-N_HALO_DEFAULT, N_HALO_DEFAULT:-N_HALO_DEFAULT, 1] = lat_fort
        metric_terms.grid.data[:] = new_grid
        metric_terms._init_agrid()
        print("modified_grid")


    grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(metric_terms)
    dycore_state = fv3core.DycoreState.init_zeros(quantity_factory=quantity_factory)
    backend_uses_gpu = is_gpu_backend(dycore_state.u.metadata.gt4py_backend)

    if fortran_data is True:
        dycore_state = _overwrite_state_from_fortran_restart(
            path,
            communicator,
            dycore_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )
    else:
        dycore_state = _overwrite_state_from_restart(
            path,
            rank,
            dycore_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )
    active_packages = ["microphysics"]
    physics_state = pace.physics.PhysicsState.init_zeros(
        quantity_factory=quantity_factory, active_packages=active_packages
    )

    # if fortran_data is True:
    #     physics_state = _overwrite_state_from_fortran_restart(
    #         path,
    #         communicator,
    #         physics_state,
    #         "restart_dycore_state",
    #         backend_uses_gpu,
    #     )
    # else:
    #     physics_state = _overwrite_state_from_restart(
    #         path,
    #         rank,
    #         physics_state,
    #         "restart_physics_state",
    #         backend_uses_gpu,
    #     )

    physics_state.__post_init__(quantity_factory, active_packages)
    tendency_state = TendencyState.init_zeros(
        quantity_factory=quantity_factory,
    )

    return DriverState(
        dycore_state=dycore_state,
        physics_state=physics_state,
        tendency_state=tendency_state,
        grid_data=grid_data,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
    )
