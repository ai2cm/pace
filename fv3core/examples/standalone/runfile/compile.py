#!/usr/bin/env python3
import contextlib
import os
import sys
from argparse import ArgumentParser, Namespace

import f90nml

import pace.dsl.stencil  # noqa: F401
from fv3core import DynamicalCore
from fv3core._config import DynamicalCoreConfig
from fv3core.utils.null_comm import NullComm


local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, local)
from runfile.dynamics import get_experiment_info, setup_dycore  # noqa: E402


@contextlib.contextmanager
def no_lagrangian_contributions(dynamical_core: DynamicalCore):
    # TODO: lagrangian contributions currently cause an out of bounds iteration
    # when halo updates are disabled. Fix that bug and remove this decorator.
    # Probably requires an update to gt4py (currently v36).
    def do_nothing(*args, **kwargs):
        pass

    original_attributes = {}
    for obj in (
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_pt,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_u,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_v,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_w,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
    ):
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    for (
        obj
    ) in dynamical_core._lagrangian_to_eulerian_obj._mapn_tracer._list_of_remap_objects:
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    try:
        yield
    finally:
        for obj, original in original_attributes.items():
            obj._lagrangian_contributions = original


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
        dycore, dycore_args = setup_dycore(
            dycore_config, mpi_comm, args.backend, is_baroclinic_test_case
        )
        with no_lagrangian_contributions(dynamical_core=dycore):
            dycore.step_dynamics(**dycore_args)
        print("SUCCESS")
