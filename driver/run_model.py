import sys

sys.path.append("/usr/local/serialbox/python/")
import serialbox
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
from fv3gfs.physics.stencils.physics import Physics
from model_state import ModelState, DycoreState
from fv3gfs.util.constants import N_HALO_DEFAULT
from fv3core.grid import MetricTerms
from baroclinic_initialization import InitBaroclinic
# TODO move to utils
#from fv3core.utils.global_constants import LON_OR_LAT_DIM
LON_OR_LAT_DIM = "lon_or_lat"
sys.path.append("/port_dev/tests/savepoint/translate/")
from translate_update_dwind_phys import TranslateUpdateDWindsPhys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
backend="numpy"
fv3core.set_backend(backend)
fv3core.set_rebuild(False)
fv3core.set_validate_args(False)
case_name = "/port_dev/fv3core/test_data/c12_6ranks_baroclinic_dycore_microphysics"
#case_name = "c12_6ranks_baroclinic_dycore_microphysics"
spec.set_namelist(case_name + "/input.nml")
namelist = spec.namelist
experiment_name = yaml.safe_load(open(case_name + "/input.yml", "r",))[
    "experiment_name"
]

# set up of helper structures
serializer = serialbox.Serializer(
    serialbox.OpenModeKind.Read, case_name, "Generator_rank" + str(rank),
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
layout = namelist.layout
partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
communicator = util.CubedSphereCommunicator(comm, partitioner)


sizer = util.SubtileGridSizer.from_tile_params(
            nx_tile=namelist.npx - 1,
            ny_tile=namelist.npy - 1,
            nz=namelist.npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
            },
            layout=layout,
        )
quantity_factory = util.QuantityFactory.from_backend(
    sizer, backend=backend
)
# TODO use MetricTerms created grid for grid_data
new_grid = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)


# read in missing grid info for physics - this will be removed
dwind = TranslateUpdateDWindsPhys(grid)
missing_grid_info = dwind.collect_input_data(
    serializer, serializer.get_savepoint("FVUpdatePhys-In")[0]
)
init_mode = 'serialized_data'

if init_mode == 'serialized_data':
    # initialize from serialized data
    state = ModelState.init_from_serialized_data(serializer, grid, quantity_factory,  namelist, communicator, missing_grid_info)
if init_mode == 'empty':
    state = ModelState.init_empty(grid, quantity_factory, namelist, communicator, missing_grid_info)
if init_mode == 'baroclinic_dataclass':
    # baroclinic intialization option
    dycore_state = InitBaroclinic.init_empty() # e.g. class InitBaroclinic(DycoreState)
    dycore_state.baroclinic_initialization()
    state = ModelState.init_from_dycore_state(dycore_state, grid, quantity_factory, namelist, communicator, missing_grid_info)
if init_mode == 'baroclinic_numpy':
    # baroclinic intialization option
    numpy_arrays = compute_baroclinic_initialization(fields(DycoreState))
    state = ModelState.init_from_numpy_arrays(numpy_arrays, grid, quantity_factory, namelist, communicator, missing_grid_info)
# TODO
do_adiabatic_init = False
# TODO derive from namelist
bdt = 225.0
# initialize dynamical core and physics objects
dycore = fv3core.DynamicalCore(
    comm=communicator,
    grid_data=grid.grid_data, # TODO derive from new_grid
    grid_indexing=grid.grid_indexing,
    damping_coefficients=grid.damping_coefficients,
    config=namelist.dynamical_core,
    ak=new_grid.ak,
    bk=new_grid.bk,
    phis=state.dycore_state.phis_quantity, #state["surface_geopotential"],
    ptop = new_grid.ptop,
    ks = new_grid.ks
)

physics = Physics(grid, namelist, 300.0)#new_grid.ptop)

for t in range(1, 10):
    state.step_dynamics(dycore, do_adiabatic_init, bdt)
    state.step_physics(physics)

    if t % 5 == 0:
        comm.Barrier()
        output_vars = [
            "u",
            "v",
            "ua",
            "va",
            "pt",
            "delp",
            "qvapor",
            "qliquid",
            "qice",
            "qrain",
            "qsnow",
            "qgraupel",
        ]
        output = {}

        for key in output_vars:
            getattr(state.dycore_state, key).synchronize()
            output[key] = np.asarray(getattr(state.dycore_state, key))
        np.save("pace_output_t_" + str(t) + "_rank_" + str(rank) + ".npy", output)

