import sys

sys.path.append("/usr/local/serialbox/python/")
import serialbox
import copy
import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import yaml
from mpi4py import MPI

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
import fv3gfs.util as util
import gt4py

from fv3gfs.physics.physics_state import PhysicsState
from fv3gfs.physics.stencils.microphysics import Microphysics
from fv3gfs.physics.stencils.physics import Physics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
fv3core.set_backend("numpy")
fv3core.set_rebuild(False)
fv3core.set_validate_args(False)
global_config.set_do_halo_exchange(True)

spec.set_namelist("c12_6ranks_baroclinic_dycore_microphysics_day_10/input.nml")

experiment_name = yaml.safe_load(
    open(
        "c12_6ranks_baroclinic_dycore_microphysics_day_10/input.yml",
        "r",
    )
)["experiment_name"]

# set up of helper structures
serializer = serialbox.Serializer(
    serialbox.OpenModeKind.Read,
    "c12_6ranks_baroclinic_dycore_microphysics_day_10",
    "Generator_rank" + str(rank),
)
cube_comm = util.CubedSphereCommunicator(
    comm,
    util.CubedSpherePartitioner(util.TilePartitioner(spec.namelist.layout)),
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

# set up grid-dependent helper structures
layout = spec.namelist.layout
partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
communicator = util.CubedSphereCommunicator(comm, partitioner)

# create a state from serialized data
savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
driver_object = fv3core.testing.TranslateFVDynamics([grid])
input_data = driver_object.collect_input_data(serializer, savepoint_in)
input_data["comm"] = communicator
state = driver_object.state_from_inputs(input_data)
dycore = fv3core.DynamicalCore(
    communicator,
    spec.namelist,
    state["atmosphere_hybrid_a_coordinate"],
    state["atmosphere_hybrid_b_coordinate"],
    state["surface_geopotential"],
)

dycore.step_dynamics(
    state,
    input_data["consv_te"],
    input_data["do_adiabatic_init"],
    input_data["bdt"],
    input_data["ptop"],
    input_data["n_split"],
    input_data["ks"],
)

phy = Physics(grid, spec.namelist)
phy(state, rank)
