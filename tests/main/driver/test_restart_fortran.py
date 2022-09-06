import numpy as np
import os

from datetime import datetime
from gt4py.storage.storage import CPUStorage
from mpi4py import MPI
from netCDF4 import Dataset
from pace.driver import DriverState
#from pace.driver.initialization import RestartConfig
from pace.util import CubedSphereCommunicator, CubedSpherePartitioner, Quantity, QuantityFactory, SubtileGridSizer, TilePartitioner
#from pace.util._properties import RESTART_PROPERTIES, RestartProperties
from typing import Tuple, Union

import importlib
import pace.util._properties
importlib.reload(pace.util._properties)
from pace.util._properties import RESTART_PROPERTIES, RestartProperties

import pace.driver.initialization
importlib.reload(pace.driver.initialization)
from pace.driver.initialization import RestartConfig

nx_tile = 12
ny_tile = 12
nz = 79
n_halo = 3
layout = (1, 1)
backend = "numpy"

fortran_data = True
restart_path = "/home/ajdas/pace/restart_data/v4.0"
restart_time = datetime.strptime("2016-08-11 00:00:00", "%Y-%m-%d %H:%M:%S")


dimensions_dict = {
    "nx_tile": nx_tile,
    "ny_tile": ny_tile,
    "nz": nz,
    "n_halo": n_halo
}

restart_dict = {
    "fortran_data": fortran_data,
    "restart_path": restart_path,
    "restart_time": restart_time,
}


def initialize_driver(layout: tuple = layout, backend: str = backend, dimensions: dict = dimensions_dict, restart: dict = restart_dict) -> DriverState:

    mpi_comm = MPI.COMM_WORLD

    rank = mpi_comm.Get_rank()

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=dimensions["nx_tile"],
        ny_tile=dimensions["ny_tile"],
        nz=dimensions["nz"],
        n_halo=dimensions["n_halo"],
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    restart_config = RestartConfig(path=restart["restart_path"], start_time=restart["restart_time"], fortran_data=restart["fortran_data"])
    driver = restart_config.get_driver_state(quantity_factory, communicator)

    return driver, rank


def test_fortran_driver():

    driver, rank = initialize_driver(layout, backend, dimensions_dict, restart_dict)
    state_type = {
        "dycore": driver.dycore_state, 
    }
    #     "tendency": driver.tendency_state, 
    #     "grid_data": driver.grid_data, 
    #     "damping_coefficients": driver.damping_coefficients, 
    #     "driver_grid_data": driver.driver_grid_data,
    #     "physics": driver.physics_state, 
    # }

    state_message = {}
    for state in state_type.keys():

        message = _test_fortran_component_state(state_type[state], rank)
        state_message[state] = message
    
    return state_message


def _test_fortran_component_state(state, rank):

    field_in_state, _ = _retrieve_field_in_state(state)

    message = []
    for field in field_in_state:
        source_file, _ = _check_field_has_source(field)

        var_state = state.__dict__[field]

        if not source_file:
            message.append(field + " - skipped" + " ... shape " + str(var_state.data.shape))
        
        else:
            var_restart = _fetch_source_restart(field, rank)
            if isinstance(var_state, Quantity):
                var_state = var_state.view[:]
            elif isinstance(var_state, CPUStorage):
                nhalo = dimensions_dict["n_halo"]
                var_state = var_state.data[nhalo:-nhalo-1, nhalo:-nhalo-1, :-1]
            
            np.testing.assert_array_equal(var_state, var_restart)
            message.append(field + " - PASSED" + " ... shape " + str(var_state.data.shape))
    
    return message


def _retrieve_field_in_state(driver_state: DriverState) -> Tuple[list, list]:
    
    variable_dict = driver_state.__dict__
    fields = list(variable_dict.keys())
    
    field_in_state = []
    other_in_state = []
    for field in fields:
        if isinstance(variable_dict[field], Quantity):
            field_in_state.append(field)
        elif isinstance(variable_dict[field], CPUStorage):
            field_in_state.append(field)
        else:
            other_in_state.append(field)
    
    return field_in_state, other_in_state


def _check_field_has_source(field: str, restart_properties: RestartProperties = RESTART_PROPERTIES) -> Tuple[Union[str, None], Union[str, None]]:
    
    source_file = None
    source_name = None

    for key in restart_properties.keys():
        if restart_properties[key]["driver_name"] == field:
            if "restart_file" in restart_properties[key]:
                source_file = restart_properties[key]["restart_file"]
                source_name = restart_properties[key]["restart_name"]

    return source_file, source_name


def _fetch_source_restart(field: str, rank: int, restart_properties: RestartProperties = RESTART_PROPERTIES, restart_dict: dict = restart_dict) -> np.ndarray:

    source_file, source_name = _check_field_has_source(field)
    file_name = "%s/%s%s.nc" % (restart_dict["restart_path"], source_file, rank+1)
    var = None
    if not os.path.isfile(file_name):
        print("%s does not exist." % file_name)
    else:
        data = Dataset(file_name, "r")
        var = np.squeeze(np.array(data[source_name]))
        data.close()

        if len(var.shape) == 2:
            var = np.transpose(var, (1, 0))
        elif len(var.shape) == 3:
            var = np.transpose(var, (2, 1, 0))

    return var


def write_restart_fortran_after_init(driver_state: DriverState, rank: int, file_name: str) -> None:
    
    field_in_state, _ = _retrieve_field_in_state(driver_state)

    vars = []
    for field in field_in_state:
        source_file, _ = _check_field_has_source(field)

        var_state = driver_state.__dict__[field]

        if source_file:

            if isinstance(var_state, Quantity):
                var_state = var_state.view[:]
            elif isinstance(var_state, CPUStorage):
                nhalo = dimensions_dict["n_halo"]                
                var_state = var_state.data[nhalo:-nhalo-1, nhalo:-nhalo-1, :-1]
        vars.append(var_state)
    
    data = Dataset(file_name + "%s.nc" % (rank+1), "w")
    data.createDimension("z", 79)
    data.createDimension("y", 12)
    data.createDimension("x", 12)
    data.createDimension("y_interface", 13)
    data.createDimension("x_interface", 13)

    v0 = data.createVariable("u", "f8", ("x", "y_interface", "z"))
    v0[:] = vars[field_in_state.index("u")]

    v0 = data.createVariable("v", "f8", ("x_interface", "y", "z"))
    v0[:] = vars[field_in_state.index("v")]

    v0 = data.createVariable("W", "f8", ("x", "y", "z"))
    v0[:] = vars[field_in_state.index("w")]

    v0 = data.createVariable("T", "f8", ("x", "y", "z"))
    v0[:] = vars[field_in_state.index("pt")]

    v0 = data.createVariable("delp", "f8", ("x", "y", "z"))
    v0[:] = vars[field_in_state.index("delp")]

    v0 = data.createVariable("DZ", "f8", ("x", "y", "z"))
    v0[:] = vars[field_in_state.index("delz")]

    v0 = data.createVariable("phis", "f8", ("x", "y"))
    v0[:] = vars[field_in_state.index("phis")]


    data.close()



    return

if __name__ == "__main__":

    message = test_fortran_driver()
    
    for key in message.keys():
        print()
        print(key)
        print("\n".join(message[key]))
