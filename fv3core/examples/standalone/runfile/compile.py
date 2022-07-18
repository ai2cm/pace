#!/usr/bin/env python3
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace

import f90nml

import pace.dsl.stencil  # noqa: F401
from fv3core._config import DynamicalCoreConfig
from pace.util.null_comm import NullComm
import gt4py.config
import math

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, local)
from runfile.dynamics import get_experiment_info, setup_dycore  # noqa: E402


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "backend", type=str, action="store", help="gt4py backend to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    namelist = f90nml.read(args.data_dir + "/input.nml")
    experiment_name, is_baroclinic_test_case = get_experiment_info(args.data_dir)
    dycore_config = DynamicalCoreConfig.from_f90nml(namelist)
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    compile_steps = dycore_config.layout[0] * dycore_config.layout[1]
    iterations = math.ceil(compile_steps / size)

    for tile in range(iterations):
        target_rank = rank + size * tile
        mpi_comm = NullComm(
            rank=target_rank,
            total_ranks=6 * dycore_config.layout[0] * dycore_config.layout[1],
            fill_value=0.0,
        )
        gt4py.config.cache_settings["root_path"] = os.environ.get(
            "GT_CACHE_DIR_NAME", "."
        )
        gt4py.config.cache_settings["dir_name"] = os.environ.get(
            "GT_CACHE_ROOT", f".gt_cache_{mpi_comm.Get_rank():06}"
        )
        dycore, dycore_args, stencil_factory = setup_dycore(
            dycore_config,
            mpi_comm,
            args.backend,
            is_baroclinic_test_case,
            args.data_dir,
        )
    # NOTE (jdahm): Temporary until driver initialization-based cache is merged
    root_path = gt4py.config.cache_settings["root_path"]
    tile_size = dycore_config.layout[0] * dycore_config.layout[1]
    for rank in range(tile_size):
        for tile in range(1, 6):
            shutil.copytree(
                f"{root_path}/.gt_cache_{rank:06}",
                f"{root_path}/.gt_cache_{rank + tile*tile_size:06}",
                dirs_exist_ok=True,
            )
    print("SUCCESS")
