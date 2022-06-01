#!/usr/bin/env python3

import os.path
import sys

import f90nml

import fv3core
import fv3gfs.physics
import pace.dsl
from pace.driver import BaroclinicConfig, Driver, DriverConfig, NullCommConfig
from pace.util import Namelist


def initialize_caches(
    namelist: Namelist, backend: str, rank: int, num_ranks: int
) -> Driver:
    driver_config = DriverConfig(
        stencil_config=pace.dsl.StencilConfig(
            backend=backend, rebuild=True, validate_args=False, format_source=False
        ),
        initialization=BaroclinicConfig(),
        nx_tile=namelist.npx - 1,
        nz=namelist.npz,
        layout=namelist.layout,
        dt_atmos=namelist.dt_atmos,
        dycore_config=fv3core.DynamicalCoreConfig.from_namelist(namelist),
        physics_config=fv3gfs.physics.PhysicsConfig.from_namelist(namelist),
        comm_config=NullCommConfig(rank=rank, total_ranks=num_ranks),
    )

    return Driver(config=driver_config)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: initialize_caches.py driver_data_path backend")

    driver_data_path, backend = sys.argv[1], sys.argv[2]

    namelist = Namelist.from_f90nml(
        f90nml.read(os.path.join(driver_data_path, "input.nml"))
    )

    num_ranks = namelist.layout[0] * namelist.layout[1] * 6

    for rank in range(num_ranks):
        initialize_caches(namelist, backend, rank, num_ranks)
