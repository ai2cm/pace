import dataclasses
from dataclasses import fields
from typing import Union

import xarray as xr

import fv3core
import fv3gfs.physics
import pace.util
import pace.util.grid
from pace.util.grid import DampingCoefficients


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
    physics_state: fv3gfs.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData

    @classmethod
    def load_state_from_restart(
        cls,
        restart_path: str,
        driver_config,
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
            sizer, backend=driver_config.stencil_config.backend
        )
        state = _restart_driver_state(
            restart_path, communicator.rank, quantity_factory, communicator
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


def _overwrite_state_from_restart(
    path: str,
    rank: int,
    state: Union[fv3core.DycoreState, fv3gfs.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
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
            state.__dict__[_field.name].data[:] = df[_field.name].data[:]
    return state


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: pace.util.QuantityFactory,
    communicator: pace.util.CubedSphereCommunicator,
):
    metric_terms = pace.util.grid.MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )
    grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(metric_terms)
    dycore_state = fv3core.DycoreState.init_zeros(quantity_factory=quantity_factory)
    dycore_state = _overwrite_state_from_restart(
        path, rank, dycore_state, "restart_dycore_state"
    )
    active_packages = ["microphysics"]
    physics_state = fv3gfs.physics.PhysicsState.init_zeros(
        quantity_factory=quantity_factory, active_packages=active_packages
    )
    physics_state = _overwrite_state_from_restart(
        path, rank, physics_state, "restart_physics_state"
    )
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
