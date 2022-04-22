import time

import click
import f90nml

# Use this to add non-installed seriablox path
# import sys
# sys.path.append("/usr/local/serialbox/python/")
import numpy as np
import serialbox
from mpi4py import MPI

import fv3core
import fv3core.testing
import pace.dsl
import pace.stencils.testing
import pace.util as util
from fv3core import DynamicalCoreConfig
from fv3gfs.physics import PhysicsConfig
from fv3gfs.physics.stencils.physics import Physics
from pace.stencils.testing.grid import Grid
from pace.util import Namelist
from pace.util.grid import DampingCoefficients, DriverGridData, GridData, MetricTerms


MODEL_OUT_DIR = "./model_output"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def print_for_rank0(msg: str):
    """Only prints when rank is 0. Flush immediately."""
    if rank == 0:
        print(f"[R{rank}]{msg}", flush=True)


class DeactivatedDycore:
    def __init__(self) -> None:
        pass

    def step_dynamics(*args, **kwargs):
        pass


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=True, default="10")
@click.argument("backend", required=True, default="numpy")
@click.option("--run-dycore/--skip-dycore", default=True)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    run_dycore: bool,
):
    print_for_rank0(f"Running {data_directory} on {backend}")
    print_for_rank0("Init & timestep 0")
    start = 0
    if rank == 0:
        start = time.time()

    f90_namelist = f90nml.read(data_directory + "/input.nml")
    namelist = Namelist.from_f90nml(f90_namelist)
    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )

    # get grid object with indices used for translating from serialized data
    grid = Grid.from_namelist(namelist, rank, backend)

    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    stencil_factory = pace.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid.grid_indexing,
    )

    # set up domain decomposition
    layout = namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    metric_terms = MetricTerms.from_tile_sizing(
        npx=namelist.npx,
        npy=namelist.npy,
        npz=namelist.npz,
        communicator=communicator,
        backend=backend,
    )

    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    driver_object = fv3core.testing.TranslateFVDynamics(
        [grid], namelist, stencil_factory
    )
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    input_data["comm"] = communicator
    state = driver_object.state_from_inputs(input_data)

    dwind = DriverGridData.new_from_metric_terms(metric_terms)
    grid_data = GridData.new_from_metric_terms(metric_terms)
    # initialize dynamical core and physics objects
    dycore_config = DynamicalCoreConfig.from_namelist(namelist)
    if run_dycore:
        dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=stencil_factory,
            damping_coefficients=DampingCoefficients.new_from_metric_terms(
                metric_terms
            ),
            config=dycore_config,
            phis=state.phis_quantity,
            state=state,
        )
    else:
        dycore = DeactivatedDycore()
    physics_config = PhysicsConfig.from_namelist(namelist)
    step_physics = Physics(
        stencil_factory=stencil_factory,
        grid_data=grid_data,
        namelist=physics_config,
    )
    # TODO include functionality that uses and changes this
    do_adiabatic_init = False
    # TODO compute from namelist
    bdt = 225.0

    print_for_rank0(f"Init & timestep 0 done in {time.time() - start}s ")

    for t in range(1, int(time_steps) + 1):
        if rank == 0:
            start = time.time()
        dycore.step_dynamics(
            state,
            namelist.consv_te,
            do_adiabatic_init,
            bdt,
            namelist.n_split,
        )
        step_physics(state, 300.0)
        if t % 5 == 0:
            io_start = 0
            if rank == 0:
                io_start = time.time()
            comm.Barrier()
            output_vars = [
                "ua",
                "qrain",
            ]
            output = {}

            for key in output_vars:
                state[key].synchronize()
                output[key] = np.asarray(state[key])
            np.savez_compressed(
                f"{MODEL_OUT_DIR}/pace_output_t_{str(t)}_rank_{str(rank)}.npz",
                output,
            )
            print_for_rank0(f"I/O at timestep {t} done in {time.time() - io_start}s")
        else:
            print_for_rank0(f"Timestep {t} done in {time.time() - start}s")


if __name__ == "__main__":
    # Make sure the model output directory exists
    from pathlib import Path

    Path(MODEL_OUT_DIR).mkdir(parents=True, exist_ok=True)
    # Run the experiment
    driver()
