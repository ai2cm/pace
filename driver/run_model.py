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
from model_state import ModelState
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
#new_grid = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)

## create a state from serialized data
#savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
#driver_object = fv3core.testing.TranslateFVDynamics([grid])
#input_data = driver_object.collect_input_data(serializer, savepoint_in)
#input_data["comm"] = communicator
#state = driver_object.state_from_inputs(input_data)

# read in missing grid info for physics - this will be removed
dwind = TranslateUpdateDWindsPhys(grid)
missing_grid_info = dwind.collect_input_data(
    serializer, serializer.get_savepoint("FVUpdatePhys-In")[0]
)
init_mode = 'serialized_data'
if init_mode == 'baroclinic':
    # baroclinic intialization option
    dycore_state = InitBaroclinic.init_empty()
    dycore_state.baroclinic_initiaization()#new_grid.ak, new_grid.bk)
    state = ModelState.init_from_dycore_state(dycore_state, grid, quantity_factory, namelist, communicator, missing_grid_info)
if init_mode == 'serialized_data':
    # initialize from serialized data
    state = ModelState.init_from_serialized_data(serializer, grid, quantity_factory,  namelist, communicator, missing_grid_info)
if init_mode == 'empty':
    state = ModelState.init_empty(grid, quantity_factory, namelist, communicator, missing_grid_info)
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
    ak=state.dycore_state.ak_quantity,#new_grid.ak, #state["atmosphere_hybrid_a_coordinate"],
    bk=state.dycore_state.bk_quantity, #new_grid.bk, #state["atmosphere_hybrid_b_coordinate"],
    phis=state.dycore_state.phis_quantity, #state["surface_geopotential"],
    ptop = 300.0, #new_grid.ptop,
    ks = 18, #new_grid.ks
)

step_physics = Physics(grid, namelist, 300.0)#new_grid.ptop)

for t in range(1, 10):
    dycore.step_dynamics(
        state.dycore_state,
        do_adiabatic_init,
        bdt,  
    )
    state.update_physics_inputs_state()
    step_physics(state.physics_state)
    state.update_dycore_state()
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

