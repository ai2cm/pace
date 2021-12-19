import sys

import f90nml
import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import yaml
from mpi4py import MPI

import fv3core
import fv3core._config as spec

import pace.util as util
import gt4py

from driver_state import DriverState
from fv3core.initialization.dycore_state import DycoreState
import fv3core.initialization.baroclinic as baroclinic_init
from fv3gfs.physics import PhysicsState
from fv3gfs.physics.stencils.physics import Physics
from pace.util.constants import N_HALO_DEFAULT
from pace.util.grid import MetricTerms
from pace.stencils.testing.grid import DampingCoefficients, GridData, DriverGridData
import pace.util
import pace.dsl
import fv3core._config
from fv3core.testing.translate_fvdynamics import init_dycore_state_from_serialized_data

# TODO remove when using MetricTerms for Physics grid variables
sys.path.append("/port_dev/fv3gfs-physics/tests/savepoint/translate/")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
backend="numpy"

case_name = "/port_dev/fv3core/test_data/c12_6ranks_baroclinic_dycore_microphysics"

experiment_name = yaml.safe_load(open(case_name + "/input.yml", "r",))[
    "experiment_name"
]
namelist = fv3core._config.Namelist.from_f90nml(f90nml.read(case_name + "/input.nml"))
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

# set up domain decomposition
layout = namelist.layout
partitioner = pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout))
communicator = pace.util.CubedSphereCommunicator(comm, partitioner)

stencil_config = pace.dsl.stencil.StencilConfig(
    backend=backend,
    rebuild=False,
    validate_args=True,
)
sizer = pace.util.SubtileGridSizer.from_tile_params(
    nx_tile=namelist.npx - 1,
    ny_tile=namelist.npy - 1,
    nz=namelist.npz,
    n_halo=N_HALO_DEFAULT,
    extra_dim_lengths={},
    layout=layout,
    tile_partitioner=partitioner.tile,
    tile_rank=communicator.tile.rank,
)
quantity_factory = pace.util.QuantityFactory.from_backend(
    sizer, backend=backend
)
grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
    sizer=sizer, cube=communicator
)
stencil_factory = pace.dsl.stencil.StencilFactory(
    config=stencil_config,
    grid_indexing=grid_indexing,
)
metric_terms = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)
grid_data = GridData.new_from_metric_terms(metric_terms)
driver_grid_data = DriverGridData.new_from_metric_terms(metric_terms)

init_mode = 'serialized_data'

if init_mode == 'serialized_data':
    sys.path.append("/usr/local/serialbox/python/")
    import serialbox
    serializer = serialbox.Serializer(
    serialbox.OpenModeKind.Read, case_name, "Generator_rank" + str(rank),
    )
    dycore_state = init_dycore_state_from_serialized_data(serializer=serializer, rank=rank, backend=backend, namelist=namelist, quantity_factory=quantity_factory, stencil_factory=stencil_factory)

if init_mode == 'baroclinic':
    # create an initial state from the Jablonowski & Williamson Baroclinic
    # test case perturbation. JRMS2006

    dycore_state = baroclinic_init.init_baroclinic_state(
        metric_terms,
        adiabatic=namelist.adiabatic,
        hydrostatic=namelist.hydrostatic,
        moist_phys=namelist.moist_phys,
        comm=communicator,
    )
driver_state = DriverState.init_from_dycore_state(dycore_state,  quantity_factory, namelist, comm, grid_info=driver_grid_data)
# TODO
do_adiabatic_init = False
# TODO derive from namelist
bdt = 225.0
# initialize dynamical core and physics objects
dycore = fv3core.DynamicalCore(
    comm=communicator,
    grid_data=grid_data,
    stencil_factory=stencil_factory,
    damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
    config=namelist.dynamical_core,
    phis=dycore_state.phis, 
)

step_physics = Physics(stencil_factory=stencil_factory,   grid_data=grid_data, namelist=namelist,comm=communicator, rank=rank, grid_info=driver_grid_data, ptop=metric_terms.ptop)

for t in range(1, 2):
    dycore.step_dynamics(
        state=driver_state.dycore_state,
        conserve_total_energy=namelist.consv_te,
        n_split=namelist.n_split,
        do_adiabatic_init=do_adiabatic_init,
        timestep=bdt,  
    )
    #driver.update_physics_inputs_state(driver_state.dycore_state, driver_state.physics_state)
    step_physics(driver_state.dycore_state, driver_state.physics_state)
    #driver.apply_physics_to_dynamics(driver_state.physics_state, driver_state.dycore_state)
    if t % 5 == 0:
        comm.Barrier()
       
        output = {}

        for key in output_vars:
            getattr(driver_state.dycore_state, key).synchronize()
            output[key] = np.asarray(getattr(driver_state.dycore_state, key))
        np.save("pace_output_t_" + str(t) + "_rank_" + str(rank) + ".npy", output)

