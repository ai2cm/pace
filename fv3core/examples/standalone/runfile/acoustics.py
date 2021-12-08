#!/usr/bin/env python3
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import click
import serialbox
import yaml
from timing import collect_data_and_write_to_file

import fv3core
import fv3core._config as spec
import fv3core.testing
import pace.util as util
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.utils.grid import Grid
from fv3core.utils.null_comm import NullComm


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def set_up_namelist(data_directory: str) -> None:
    """
    Reads the namelist at the given directory and sets
    the global fv3core config to it
    """
    spec.set_namelist(data_directory + "/input.nml")


def initialize_serializer(data_directory: str, rank: int = 0) -> serialbox.Serializer:
    """Creates a Serializer based on the data-directory and the rank"""
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )


def initialize_fv3core(backend: str, disable_halo_exchange: bool) -> None:
    """
    Initializes globalfv3core config to the arguments for single runs
    with the given backend and choice of halo updates
    """
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)


def read_input_data(grid: Grid, serializer: serialbox.Serializer) -> Dict[str, Any]:
    """Uses the serializer to read the input data from disk"""
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(
    grid: Grid, input_data: Dict[str, Any]
) -> Dict[str, SimpleNamespace]:
    """
    Transforms the input data from the dictionary of strings
    to arrays into a state  we can pass in

    Input is a dict of arrays. These are transformed into Storage arrays
    useable in GT4Py

    This will also take care of reshaping the arrays into same sized
    fields as required by the acoustics
    """
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return {"state": statevars}


def set_up_communicator(
    disable_halo_exchange: bool,
) -> Tuple[Optional[MPI.Comm], Optional[util.CubedSphereCommunicator]]:
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    if MPI is not None:
        comm = MPI.COMM_WORLD
    else:
        comm = None
    if not disable_halo_exchange:
        assert comm is not None
        cube_comm = util.CubedSphereCommunicator(comm, partitioner)
    else:
        cube_comm = util.CubedSphereCommunicator(NullComm(0, 0), partitioner)
    return comm, cube_comm


def get_experiment_name(
    data_directory: str,
) -> str:
    return yaml.safe_load(
        open(
            data_directory + "/input.yml",
            "r",
        )
    )["experiment_name"]


def initialize_timers() -> Tuple[util.Timer, util.Timer, List, List]:
    total_timer = util.Timer()
    total_timer.start("total")
    timestep_timer = util.Timer()
    return total_timer, timestep_timer, [], []


def read_and_reset_timer(timestep_timer, times_per_step, hits_per_step):
    times_per_step.append(timestep_timer.times)
    hits_per_step.append(timestep_timer.hits)
    timestep_timer.reset()
    return times_per_step, hits_per_step


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1")
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--disable_halo_exchange/--no-disable_halo_exchange", default=False)
@click.option("--print_timings/--no-print_timings", default=True)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    disable_halo_exchange: bool,
    print_timings: bool,
):
    total_timer, timestep_timer, times_per_step, hits_per_step = initialize_timers()
    with total_timer.clock("initialization"):
        set_up_namelist(data_directory)
        serializer = initialize_serializer(data_directory)
        initialize_fv3core(backend, disable_halo_exchange)
        mpi_comm, communicator = set_up_communicator(disable_halo_exchange)
        grid = spec.make_grid_with_data_from_namelist(
            spec.namelist, communicator, backend
        )
        spec.set_grid(grid)

        input_data = read_input_data(grid, serializer)
        experiment_name = get_experiment_name(data_directory)
        acoustics_object = AcousticDynamics(
            communicator,
            grid.stencil_factory,
            grid.grid_data,
            grid.damping_coefficients,
            grid.grid_type,
            grid.nested,
            grid.stretched_grid,
            spec.namelist.dynamical_core.acoustic_dynamics,
            input_data["pfull"],
            input_data["phis"],
        )

        state = get_state_from_input(grid, input_data)

        # warm-up timestep.
        # We're intentionally not passing the timer here to exclude
        # warmup/compilation from the internal timers
        acoustics_object(**state)

    # we set up a specific timer for each timestep
    # that is cleared after so we get individual statistics
    for _ in range(int(time_steps) - 1):
        # this loop is not required
        # but make performance numbers comparable with FVDynamics
        for _ in range(spec.namelist.k_split):
            with timestep_timer.clock("DynCore"):
                acoustics_object(**state)
        times_per_step, hits_per_step = read_and_reset_timer(
            timestep_timer, times_per_step, hits_per_step
        )
    total_timer.stop("total")
    times_per_step, hits_per_step = read_and_reset_timer(
        total_timer, times_per_step, hits_per_step
    )

    experiment_info = {
        "name": "acoustics",
        "dataset": experiment_name,
        "timesteps": time_steps,
        "backend": backend,
        "halo_update": not disable_halo_exchange,
        "hash": "",
    }
    if print_timings:
        # Collect times and output statistics in json
        collect_data_and_write_to_file(
            mpi_comm, hits_per_step, times_per_step, experiment_info
        )


if __name__ == "__main__":
    driver()
