#!/usr/bin/env python3

import copy
import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI

import fv3core.initialization.aqua_tile as aqua_tile_init
from fv3core.grid import MetricTerms
from fv3core.utils.grid import DampingCoefficients, GridData


# Dev note: the GTC toolchain fails if xarray is imported after gt4py
# fv3gfs.util imports xarray if it's available in the env.
# fv3core imports gt4py.
# To avoid future conflict creeping back we make util imported prior to
# fv3core. isort turned off to keep it that way.
# isort: off
import fv3gfs.util as util
from fv3core.utils.null_comm import NullComm

# isort: on

import fv3core
import fv3core._config as spec
import fv3core.testing


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "time_step",
        type=int,
        action="store",
        help="number of timesteps to execute",
    )
    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="gt4py backend to use",
    )
    parser.add_argument(
        "--disable_halo_exchange",
        action="store_true",
        help="enable or disable the halo exchange",
    )
    parser.add_argument(
        "--disable_json_dump",
        action="store_true",
        help="enable or disable json dump",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable performance profiling using cProfile",
    )

    return parser.parse_args()


def set_experiment_info(
    experiment_name: str, time_step: int, backend: str
) -> Dict[str, Any]:
    experiment: Dict[str, Any] = {}
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment["setup"] = {}
    experiment["setup"]["timestamp"] = dt_string
    experiment["setup"]["dataset"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["version"] = "python/" + backend
    experiment["setup"]["format_version"] = 2
    experiment["times"] = {}
    return experiment


def collect_keys_from_data(times_per_step: List[Dict[str, float]]) -> List[str]:
    """Collects all the keys in the list of dics and returns a sorted version"""
    keys = set()
    for data_point in times_per_step:
        for k, _ in data_point.items():
            keys.add(k)
    sorted_keys = list(keys)
    sorted_keys.sort()
    return sorted_keys


def gather_timing_data(
    times_per_step: List[Dict[str, float]],
    results: Dict[str, Any],
    comm: MPI.Comm,
    root: int = 0,
) -> Dict[str, Any]:
    """returns an updated version of  the results dictionary owned
    by the root node to hold data on the substeps as well as the main loop timers"""
    is_root = comm.Get_rank() == root
    keys = collect_keys_from_data(times_per_step)
    data: List[float] = []
    for timer_name in keys:
        data.clear()
        for data_point in times_per_step:
            if timer_name in data_point:
                data.append(data_point[timer_name])

        sendbuf = np.array(data)
        recvbuf = None
        if is_root:
            recvbuf = np.array([data] * comm.Get_size())
        comm.Gather(sendbuf, recvbuf, root=0)
        if is_root:
            results["times"][timer_name]["times"] = copy.deepcopy(recvbuf.tolist())
    return results


def write_global_timings(experiment: Dict[str, Any]) -> None:
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    with open(filename + ".json", "w") as outfile:
        json.dump(experiment, outfile, sort_keys=True, indent=4)


def gather_hit_counts(
    hits_per_step: List[Dict[str, int]], results: Dict[str, Any]
) -> Dict[str, Any]:
    """collects the hit count across all timers called in a program execution"""
    for data_point in hits_per_step:
        for name, value in data_point.items():
            if name not in results["times"]:
                print(name)
                results["times"][name] = {"hits": value, "times": []}
            else:
                results["times"][name]["hits"] += value
    return results


def collect_data_and_write_to_file(
    args: Namespace, comm: MPI.Comm, hits_per_step, times_per_step, experiment_name
) -> None:
    """
    collect the gathered data from all the ranks onto rank 0 and write the timing file
    """
    is_root = comm.Get_rank() == 0
    results = None
    if is_root:
        print("Gathering Times")
        results = set_experiment_info(experiment_name, args.time_step, args.backend)
        results = gather_hit_counts(hits_per_step, results)

    results = gather_timing_data(times_per_step, results, comm)

    if is_root:
        write_global_timings(results)


if __name__ == "__main__":
    timer = util.Timer()
    timer.start("total")
    with timer.clock("initialization"):
        args = parse_args()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        profiler = None
        if args.profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.disable()

        fv3core.set_backend(args.backend)
        fv3core.set_rebuild(False)
        fv3core.set_validate_args(False)

        spec.set_namelist(args.data_dir + "/input.nml")

        experiment_name = os.path.basename(os.path.normpath(args.data_dir))

        # set up of helper structures
        if args.disable_halo_exchange:
            mpi_comm = NullComm(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
        else:
            mpi_comm = MPI.COMM_WORLD

        namelist = spec.namelist
        # set up grid-dependent helper structures
        partitioner = util.TilePartitioner(namelist.layout)
        communicator = util.TileCommunicator(mpi_comm, partitioner)
        # generate the grid
        grid = spec.make_grid_with_data_from_namelist(
            namelist, communicator, args.backend
        )
        spec.set_grid(grid)

        # TODO remove this creation of the legacy grid once everything that
        # references it is updated or removed
        grid = spec.make_grid_from_namelist(namelist, communicator.rank)
        spec.set_grid(grid)

        metric_terms = MetricTerms.from_single_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=namelist.npz,
            communicator=communicator,
            backend=args.backend,
        )

        # create an initial state from the Jablonowski & Williamson Baroclinic
        # test case perturbation. JRMS2006
        state = aqua_tile_init.init_doubly_periodic_state(
            metric_terms,
            hydrostatic=namelist.hydrostatic,
            moist_phys=namelist.moist_phys,
            comm=communicator,
        )

        dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=GridData.new_from_metric_terms(metric_terms),
            stencil_factory=grid.stencil_factory,
            damping_coefficients=DampingCoefficients.new_from_metric_terms(
                metric_terms
            ),
            config=spec.namelist.dynamical_core,
            phis=state.phis_quantity,
        )
        # TODO include functionality that uses and changes this
        do_adiabatic_init = False
        # TODO compute from namelist
        bdt = 225.0

        # warm-up timestep.
        # We're intentionally not passing the timer here to exclude
        # warmup/compilation from the internal timers
        if rank == 0:
            print("timestep 1")
        dycore.step_dynamics(
            state,
            namelist.consv_te,
            do_adiabatic_init,
            bdt,
            namelist.n_split,
        )

    if profiler is not None:
        profiler.enable()

    times_per_step = []
    hits_per_step = []
    # we set up a specific timer for each timestep
    # that is cleared after so we get individual statistics
    timestep_timer = util.Timer()
    for i in range(args.time_step - 1):
        with timestep_timer.clock("mainloop"):
            if rank == 0:
                print(f"timestep {i+2}")
            dycore.step_dynamics(
                state,
                namelist.consv_te,
                do_adiabatic_init,
                bdt,
                namelist.n_split,
                timestep_timer,
            )
        times_per_step.append(timestep_timer.times)
        hits_per_step.append(timestep_timer.hits)
        timestep_timer.reset()

    if profiler is not None:
        profiler.disable()

    timer.stop("total")
    times_per_step.append(timer.times)
    hits_per_step.append(timer.hits)

    # output profiling data
    if profiler is not None:
        profiler.dump_stats(f"fv3core_{experiment_name}_{args.backend}_{rank}.prof")

    # Timings
    if not args.disable_json_dump:
        # Collect times and output statistics in json
        MPI.COMM_WORLD.Barrier()
        collect_data_and_write_to_file(
            args, MPI.COMM_WORLD, hits_per_step, times_per_step, experiment_name
        )
    else:
        # Print a brief summary of timings
        # Dev Note: we especially do _not_ gather timings here to have a
        # no-MPI-communication codepath
        print(f"Rank {rank} done. Total time: {timer.times['total']}.")

    if rank == 0:
        print("SUCCESS")
