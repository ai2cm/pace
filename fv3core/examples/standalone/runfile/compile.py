#!/usr/bin/env python3
import math
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace

import f90nml
import gt4py.cartesian.config

import pace.dsl.stencil  # noqa: F401
from pace.fv3core._config import DynamicalCoreConfig
from pace.util.null_comm import NullComm


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
        "backend",
        type=str,
        action="store",
        help="gt4py backend to use",
    )
    parser.add_argument(
        "target_dir",
        type=str,
        action="store",
        help="directory to copy the caches to",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    namelist = f90nml.read(args.data_dir + "/input.nml")
    experiment_name, is_baroclinic_test_case = get_experiment_info(args.data_dir)
    dycore_config = DynamicalCoreConfig.from_f90nml(namelist)
    if MPI is not None:
        comm = MPI.COMM_WORLD
        global_rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        global_rank = 0
        size = 1
    sub_tiles = dycore_config.layout[0] * dycore_config.layout[1]
    iterations = math.ceil(sub_tiles / size)
    gt4py.cartesian.config.cache_settings["root_path"] = os.environ.get(
        "GT_CACHE_DIR_NAME", "."
    )
    for iteration in range(iterations):
        top_tile_rank = global_rank + size * iteration
        if top_tile_rank < sub_tiles:
            mpi_comm = NullComm(
                rank=top_tile_rank,
                total_ranks=6 * sub_tiles,
                fill_value=0.0,
            )
            gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
                "GT_CACHE_ROOT", f".gt_cache_{mpi_comm.Get_rank():06}"
            )
            dycore, dycore_args, stencil_factory = setup_dycore(
                dycore_config,
                mpi_comm,
                args.backend,
                is_baroclinic_test_case,
                args.data_dir,
            )
            if stencil_factory.config.dace_config.is_dace_orchestrated():
                raise RuntimeError(
                    "cannot use a setup of the dycore to initialize "
                    "orchestrated code, the code needs to be run!"
                )
            print(f"rank {global_rank} compiled target rank {top_tile_rank}")
    # NOTE (jdahm): Temporary until driver initialization-based cache is merged
    if comm is not None:
        comm.Barrier()
    root_path = gt4py.cartesian.config.cache_settings["root_path"]
    for iter in range(iterations):
        top_tile_rank = global_rank + size * iter
        if top_tile_rank < sub_tiles:
            for tile in range(6):
                shutil.copytree(
                    f"{root_path}/.gt_cache_{top_tile_rank:06}",
                    f"{args.target_dir}/.gt_cache"
                    f"_{(top_tile_rank + tile*sub_tiles):06}",
                    dirs_exist_ok=True,
                )
                print(
                    f"rank {global_rank} copied for "
                    f"target rank {(top_tile_rank + tile*sub_tiles)}"
                )
    if comm is not None:
        comm.Barrier()
    print(f"rank {global_rank} is past the barrier, exiting")
    if global_rank == 0:
        print("SUCCESS")
