import sys

# Use this to add non-installed seriablox path
# sys.path.append("/usr/local/serialbox/python/")
import numpy as np
import serialbox
import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List

import cupy as cp
import yaml
from mpi4py import MPI

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
import fv3gfs.util as util
import gt4py
from fv3gfs.physics.stencils.physics import Physics

import click

import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def print_for_rank0(msg: str):
    '''Only prints when rank is 0. Flush immediately.''' 
    if rank == 0:
        print(f"[R{rank}]{msg}", flush=True)


# Reuse infrastructure to read in grid variables
# add path to integration test to reuse existing grid logic
sys.path.append(
    "/scratch/snx3000/fdeconic/fv3gfs-integration/tests/savepoint/translate/"
)
from translate_update_dwind_phys import TranslateUpdateDWindsPhys


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=True, default="10")
@click.argument("backend", required=True, default="numpy")
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
):
    print_for_rank0(f"Running {data_directory} on {backend}")
    print_for_rank0(f"Init & timestep 0")
    start = 0
    if rank == 0:
        start = time.time()

    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)
    spec.set_namelist(data_directory + "/input.nml")
    
    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )

    # get grid from serialized data
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = fv3core.testing.TranslateGrid(grid_data, rank).python_grid()
    spec.set_grid(grid)

    # set up domain decomposition
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    driver_object = fv3core.testing.TranslateFVDynamics([grid])
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    input_data["comm"] = communicator
    state = driver_object.state_from_inputs(input_data)

    # read in missing grid info for physics - this will be removed
    dwind = TranslateUpdateDWindsPhys(grid)
    missing_grid_info = dwind.collect_input_data(
        serializer, serializer.get_savepoint("FVUpdatePhys-In")[0]
    )

    # initialize dynamical core and physics objects
    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=spec.grid.grid_data,
        grid_indexing=spec.grid.grid_indexing,
        damping_coefficients=spec.grid.damping_coefficients,
        config=spec.namelist.dynamical_core,
        ak=state["atmosphere_hybrid_a_coordinate"],
        bk=state["atmosphere_hybrid_b_coordinate"],
        phis=state["surface_geopotential"],
    )
    step_physics = Physics(grid, spec.namelist, communicator, missing_grid_info)

    print_for_rank0(f"Init & timestep 0 done in {time.time() - start}s ")

    for t in range(1, int(time_steps)+1):
        if rank == 0:
            start = time.time()
        dycore.step_dynamics(
            state,
            input_data["consv_te"],
            input_data["do_adiabatic_init"],
            input_data["bdt"],
            input_data["ptop"],
            input_data["n_split"],
            input_data["ks"],
        )
        step_physics(state)
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
                "model_output/pace_output_t_" + str(t) + "_rank_" + str(rank) + ".npz",
                output,
            )
            print_for_rank0(f"I/O at timestep {t} done in {time.time() - io_start}s")
        else:
            print_for_rank0(f"Timestep {t} done in {time.time() - start}s")


if __name__ == "__main__":
    driver()