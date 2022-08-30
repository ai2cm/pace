import dataclasses
from dataclasses import fields
from typing import Union

import numpy as np
import xarray as xr

import fv3core
import fv3gfs.physics
import pace.dsl.gt4py_utils as gt_utils
import pace.util
import pace.util.grid
from pace.dsl.dace.dace_config import DaceConfig
from pace.dsl.gt4py_utils import is_gpu_backend
from pace.util._properties import RESTART_PROPERTIES
from pace.util.grid import DampingCoefficients
from pace.util import N_HALO_DEFAULT as halo


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
    physics_state: fv3gfs.physics.PhysicsState
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
    state: Union[fv3core.DycoreState, fv3gfs.physics.PhysicsState, TendencyState],
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
    rank: int,
    communicator: pace.util.CubedSphereCommunicator,
    state: Union[fv3core.DycoreState, fv3gfs.physics.PhysicsState, TendencyState],
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
    #print("Now in _overwrite_state_from_fortran_restart.")


    #print("Now going to run _driver_state_to_dict")
    state_dict = _driver_state_to_dict(
        state, restart_file_prefix, is_gpu_backend
    )  # do we need to transform state?
    #print("Now out of _driver_state_to_dict")
    #print("state_dict content keys:", state_dict.keys())
    #print(state_dict.keys())


    #print("Now starting pace.util.open_restart()")
    state_dict = pace.util.open_restart(path, communicator, to_state=state_dict)
    #print("I've exited pace.util.open_restart")
    #print("state_dict keys:", state_dict.keys())
    #print("")
    # for key in state.__dict__.keys():
    #     print(key, type(state.__dict__[key]))
    # print("package:", type(state))

    #print("Now starting _dict_state_to_driver_state")
    state = _dict_state_to_driver_state(
        state_dict, state, restart_file_prefix, is_gpu_backend
    )  # if we needed to transform state
    #print("Now done with _dict_state_to_driver_state.")

    #print("Finished with _overwrite_state_from_fortran_restart.")

    return state


def _driver_state_to_dict(
    driver_state: Union[
        fv3core.DycoreState, fv3gfs.physics.PhysicsState, TendencyState
    ],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Takes a Pace driver state
    and returns a dict of state quantities with their Fortran names
    """

    #print("Now in _driver_state_to_dict")

    dict_state = {}
    tmp = type(driver_state)
    #print("Type of driver state:", tmp)
    for _field in fields(type(driver_state)):
        if "units" in _field.metadata.keys():
            #print("Field name:", _field.name)
            # for key in RESTART_PROPERTIES.keys():
            #     fname = RESTART_PROPERTIES[key]["restart_name"]
            #     if fname == _field.name:
            #         print(fname, key)
            #         dict_state[fname] = driver_state.__dict__[_field.name].data[:]
                #     fortran_name = key
                    #dict_state[fortran_name] = driver_state.__dict__[_field.name].data[:]
                
                # else:
                #     fortran_name = "NOT PRESENT IN _PROPERTIES.PY"

            # print("Fortran name:", fortran_name)                    

            #fortran_name = RESTART_PROPERTIES[_field.name]["restart_name"]
            #print("Fortran_name:", fortran_name)
            if is_gpu_backend:
                if "phy" in restart_file_prefix:
                    dict_state[_field.name] = driver_state.__dict__[_field.name][:]
                else:
                    dict_state[_field.name] = driver_state.__dict__[_field.name].data[
                        :
                    ]
            else:
                dict_state[_field.name] = driver_state.__dict__[_field.name].data[:]

    return dict_state


def _dict_state_to_driver_state(
    fortran_state: dict,
    driver_state: Union[
        fv3core.DycoreState, fv3gfs.physics.PhysicsState, TendencyState
    ],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Takes a dict of state quantities with their Fortran names and a driver state
    and populates the driver state with quantities from the dict.
    """

    #print("Now in _dict_state_to_driver_state")
    #print("Fortran state:", fortran_state.keys())
    #print("Driver state:", driver_state.__dict__.keys())  
    #print("Type of driver state:", type(driver_state))  
    #print("Driver state is physics:", isinstance(driver_state, fv3gfs.physics.physics_state.PhysicsState))

    for _field in fields(type(driver_state)):
        #print("field:", _field.name)

        if "units" in _field.metadata.keys():
            #fortran_name = RESTART_PROPERTIES[_field.name]["restart_name"]
            if is_gpu_backend:
                if "phy" in restart_file_prefix:
                    driver_state.__dict__[_field.name][:] = fortran_state[_field.name]
                else:
                    driver_state.__dict__[_field.name].data[:] = fortran_state[
                        _field.name
                    ]
            else:
                #print("Exists in fortran_state:", _field.name in fortran_state)
                if _field.name in fortran_state and _field.name in driver_state.__dict__.keys():
                    #print(_field.name, "in fortran state and driver state")
                    #print("FS shape", fortran_state[_field.name].data.shape)
                    #tp_data = np.transpose(fortran_state[_field.name].data)
                    #print("transpose shape:", tp_data.shape)
                    #print("DS shape", driver_state.__dict__[_field.name].data.shape)
                    if isinstance(driver_state, fv3gfs.physics.physics_state.PhysicsState):
                        driver_state.__dict__[_field.name][halo:-halo-1, halo:-halo-1, :-1] = np.transpose(fortran_state[_field.name].data)
                    else:
                        #print("DS shape view", driver_state.__dict__[_field.name].view[:].shape)
                        driver_state.__dict__[_field.name].view[:] = np.transpose(fortran_state[_field.name].data)
    
    return driver_state


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: pace.util.QuantityFactory,
    communicator: pace.util.CubedSphereCommunicator,
    fortran_data: bool = False,
):
    metric_terms = pace.util.grid.MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )
    grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(metric_terms)
    dycore_state = fv3core.DycoreState.init_zeros(quantity_factory=quantity_factory)
    backend_uses_gpu = is_gpu_backend(dycore_state.u.metadata.gt4py_backend)


    if fortran_data is True:
        #print("Now going into _overwrite_state_from_fortran_restart for dycore")
        dycore_state = _overwrite_state_from_fortran_restart(
            path,
            rank,
            communicator,
            dycore_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )
        #print("Now out of _overwrite_state_from_fortran_restart for dycore")
    else:
        dycore_state = _overwrite_state_from_restart(
            path,
            rank,
            dycore_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )
    active_packages = ["microphysics"]
    
    physics_state = fv3gfs.physics.PhysicsState.init_zeros(
        quantity_factory=quantity_factory, active_packages=active_packages
    )    

    if fortran_data is True:
        #print("Now going into _overwrite_state_from_fortran_restart for MP (I think?)")
        physics_state = _overwrite_state_from_fortran_restart(
            path,
            rank,
            communicator,
            physics_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )

        #print("Now out of _overwrite_state_from_fortran_restart for MP (I think?)")

    else:
        physics_state = _overwrite_state_from_restart(
            path,
            rank,
            physics_state,
            "restart_physics_state",
            backend_uses_gpu,
        )

    physics_state.__post_init__(quantity_factory, active_packages)
    tendency_state = TendencyState.init_zeros(
        quantity_factory=quantity_factory,
    )

    #print("Returning DriverState")

    return DriverState(
        dycore_state=dycore_state,
        physics_state=physics_state,
        tendency_state=tendency_state,
        grid_data=grid_data,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
    )
