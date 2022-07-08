#!/usr/bin/env python3
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace

import f90nml

import pace.dsl.stencil  # noqa: F401
from fv3core._config import DynamicalCoreConfig
from pace.util.null_comm import NullComm


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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    namelist = f90nml.read(args.data_dir + "/input.nml")
    experiment_name, is_baroclinic_test_case = get_experiment_info(args.data_dir)
    dycore_config = DynamicalCoreConfig.from_f90nml(namelist)
    for rank in range(dycore_config.layout[0] * dycore_config.layout[1]):
        mpi_comm = NullComm(
            rank=rank,
            total_ranks=6 * dycore_config.layout[0] * dycore_config.layout[1],
            fill_value=0.0,
        )
        dycore, dycore_args, stencil_factory = setup_dycore(
            dycore_config,
            mpi_comm,
            args.backend,
            is_baroclinic_test_case,
            args.data_dir,
        )
    # NOTE (jdahm): Temporary until driver initialization-based cache is merged
    for rank in range(1, 6 * dycore_config.layout[0] * dycore_config.layout[1]):
        shutil.copytree(f".gt_cache_{0:06}", f".gt_cache_{rank:06}", dirs_exist_ok=True)
    print("SUCCESS")
