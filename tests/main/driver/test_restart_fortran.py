import numpy as np

from datetime import datetime
from gt4py.storage.storage import CPUStorage
from mpi4py import MPI
from netCDF4 import Dataset
from pace.driver import DriverState
from pace.driver.initialization import RestartConfig
from pace.util import CubedSphereCommunicator, CubedSpherePartitioner, Quantity, QuantityFactory, SubtileGridSizer, TilePartitioner
from pace.util._properties import RESTART_PROPERTIES, RestartProperties
from typing import Tuple, Union


nx_tile = 12
ny_tile = 12
nz = 79
n_halo = 3
layout = (1, 1)
backend = "numpy"

fortran_data = True
restart_path = "/home/ajdas/pace/restart_data/v1.0"
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


def _check_field_has_source(field: str, restart_properties: RestartProperties = RESTART_PROPERTIES) -> Union[str, None]:
    
    source_file = False

    for key in restart_properties.keys():
        if restart_properties[key]["driver_name"] == field:
            if "restart_file" in restart_properties[key]:
                #source_file = restart_properties[key]["restart_file"]
                source_file = True

    return source_file

def _fetch_source_restart(field: str, restart_properties: RestartProperties = RESTART_PROPERTIES) -> np.ndarray:


    return


def test_fortran_dycore_state():

    driver, rank = initialize_driver(layout, backend, dimensions_dict, restart_dict)
    state = driver.dycore_state

    field_in_state, _ = _retrieve_field_in_state(state)

    source = {}
    for field in field_in_state:
        source_file = _check_field_has_source(field)
        source[field] = _check_field_has_source(field)

        if source_file:




    

    
    return source


    # for field in field_in_state:
    #     print(field)

        #break

        

    # fields_in_state = list(dycore_dict.keys())
    # #for field in fields_in_state:

    # #fields_with_units = [field in fields_in_dycore_state if "units" in dycore_dict[field].keys()]



    # #return restart_properties