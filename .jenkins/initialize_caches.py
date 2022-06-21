#!/usr/bin/env python3

import os.path
import shutil
import sys

import f90nml

import fv3core
import fv3gfs.physics
import pace.dsl
from pace.driver import (
    BaroclinicConfig,
    Driver,
    DriverConfig,
    MPICommConfig,
    NullCommConfig,
)
from pace.util import Namelist


def initialize_caches(namelist: Namelist, backend: str, comm_config) -> Driver:
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
        comm_config=comm_config,
    )

    return Driver(config=driver_config)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: initialize_caches.py driver_data_path backend [serial]")

    driver_data_path, backend = sys.argv[1], sys.argv[2]

    if len(sys.argv) > 3:
        use_serial = sys.argv[3] == "serial"

    namelist = Namelist.from_f90nml(
        f90nml.read(os.path.join(driver_data_path, "input.nml"))
    )

    num_ranks_on_tile = namelist.layout[0] * namelist.layout[1]
    num_ranks = num_ranks_on_tile * 6

    if use_serial:
        for rank in range(num_ranks_on_tile):
            comm_config = NullCommConfig(rank=rank, total_ranks=num_ranks)
            initialize_caches(namelist, backend, comm_config)

        for rank in range(1, num_ranks):
            shutil.copytree(
                f".gt_cache_{0:06}", f".gt_cache_{rank:06}", dirs_exist_ok=True
            )
    else:
        comm_config = MPICommConfig()
        initialize_caches(namelist, backend, comm_config)
