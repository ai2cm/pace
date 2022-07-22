#!/usr/bin/env python3

import copy
import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List, Tuple

import f90nml
import numpy as np
import serialbox
import yaml
from mpi4py import MPI

# NOTE: we need to import dsl.stencil prior to
# pace.util, otherwise xarray precedes gt4py, causing
# very strange errors on some systems (e.g. daint)
import pace.dsl.stencil
import pace.util as util
from fv3core import DynamicalCore
from fv3core._config import DynamicalCoreConfig
from fv3core.initialization.baroclinic import init_baroclinic_state
from fv3core.initialization.dycore_state import DycoreState
from fv3core.testing import TranslateFVDynamics
from pace.dsl import StencilFactory
from pace.dsl.dace.orchestration import DaceConfig
from pace.stencils.testing.grid import Grid
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
from pace.util.null_comm import NullComm


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
        "hash",
        type=str,
        action="store",
        help="git hash to store",
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
    experiment_name: str, time_step: int, backend: str, git_hash: str
) -> Dict[str, Any]:
    experiment: Dict[str, Any] = {}
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment["setup"] = {}
    experiment["setup"]["timestamp"] = dt_string
    experiment["setup"]["dataset"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["hash"] = git_hash
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


def get_experiment_info(data_directory: str) -> Tuple[str, bool]:
    config_yml = yaml.safe_load(
        open(
            data_directory + "/input.yml",
            "r",
        )
    )
    is_baroclinic_test_case = False
    if (
        "test_case_nml" in config_yml["namelist"].keys()
        and config_yml["namelist"]["test_case_nml"]["test_case"] == 13
    ):
        is_baroclinic_test_case = True
    print(
        "Running "
        + config_yml["experiment_name"]
        + ", and using the baroclinic test case?: "
        + str(is_baroclinic_test_case)
    )
    return config_yml["experiment_name"], is_baroclinic_test_case


def read_serialized_initial_state(rank, grid, namelist, stencil_factory, data_dir):
    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_dir,
        "Generator_rank" + str(rank),
    )
    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("Driver-In")[0]
    driver_object = TranslateFVDynamics(grid, namelist, stencil_factory)
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    state = driver_object.state_from_inputs(input_data)
    return state


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
        results = set_experiment_info(
            experiment_name, args.time_step, args.backend, args.hash
        )
        results = gather_hit_counts(hits_per_step, results)

    results = gather_timing_data(times_per_step, results, comm)

    if is_root:
        write_global_timings(results)


def setup_dycore(
    dycore_config, mpi_comm, backend, is_baroclinic_test_case, data_dir
) -> Tuple[DynamicalCore, DycoreState, StencilFactory]:
    # set up grid-dependent helper structures
    partitioner = util.CubedSpherePartitioner(
        util.TilePartitioner(dycore_config.layout)
    )
    communicator = util.CubedSphereCommunicator(mpi_comm, partitioner)
    grid = Grid.from_namelist(dycore_config, mpi_comm.rank, backend)

    dace_config = DaceConfig(
        communicator,
        backend,
        tile_nx=dycore_config.npx,
        tile_nz=dycore_config.npz,
    )
    stencil_config = pace.dsl.stencil.StencilConfig(
        compilation_config=pace.dsl.stencil.CompilationConfig(
            backend=backend, rebuild=False, validate_args=False
        ),
        dace_config=dace_config,
    )
    stencil_factory = StencilFactory(
        config=stencil_config,
        grid_indexing=grid.grid_indexing,
    )
    metric_terms = MetricTerms.from_tile_sizing(
        npx=dycore_config.npx,
        npy=dycore_config.npy,
        npz=dycore_config.npz,
        communicator=communicator,
        backend=backend,
    )
    if is_baroclinic_test_case:
        # create an initial state from the Jablonowski & Williamson Baroclinic
        # test case perturbation. JRMS2006
        state = init_baroclinic_state(
            metric_terms,
            adiabatic=dycore_config.adiabatic,
            hydrostatic=dycore_config.hydrostatic,
            moist_phys=dycore_config.moist_phys,
            comm=communicator,
        )
    else:
        state = read_serialized_initial_state(
            mpi_comm.rank, grid, dycore_config, stencil_factory, data_dir
        )
    dycore = DynamicalCore(
        comm=communicator,
        grid_data=GridData.new_from_metric_terms(metric_terms),
        stencil_factory=stencil_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=dycore_config,
        phis=state.phis,
        state=state,
    )
    dycore.update_state(
        conserve_total_energy=dycore_config.consv_te,
        do_adiabatic_init=False,
        timestep=dycore_config.dt_atmos,
        n_split=dycore_config.n_split,
        state=state,
    )
    return dycore, state, stencil_factory


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

        namelist = f90nml.read(args.data_dir + "/input.nml")
        dycore_config = DynamicalCoreConfig.from_f90nml(namelist)
        experiment_name, is_baroclinic_test_case = get_experiment_info(args.data_dir)
        if args.disable_halo_exchange:
            mpi_comm = NullComm(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
        else:
            mpi_comm = MPI.COMM_WORLD
        dycore, state, stencil_factory = setup_dycore(
            dycore_config,
            mpi_comm,
            args.backend,
            is_baroclinic_test_case,
            args.data_dir,
        )

        # warm-up timestep.
        # We're intentionally not passing the timer here to exclude
        # warmup/compilation from the internal timers
        if rank == 0:
            print("timestep 1")
        dycore.step_dynamics(state, timer)

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
            dycore.step_dynamics(state, timer=timestep_timer)
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
