import os

import fv3core.initialization.baroclinic as baroclinic_init
from fv3core.grid import MetricTerms
from fv3core.utils.grid import DampingCoefficients, GridData


# Dev note: the GTC toolchain fails if xarray is imported after gt4py
# fv3gfs.util imports xarray if it's available in the env.
# fv3core imports gt4py.
# To avoid future conflict creeping back we make util imported prior to
# fv3core. isort turned off to keep it that way.
# isort: off
import fv3gfs.util
from fv3core.utils.null_comm import NullComm

# isort: on

import f90nml

import fv3core
import fv3core.testing
from fv3core._config import Namelist


DIR = os.path.abspath(os.path.dirname(__file__))
NAMELIST_FILENAME = os.path.join(DIR, "c12_namelist.nml")


def call_dycore():
    stencil_config = (
        fv3core.StencilConfig(
            backend="numpy",
            rebuild=False,
            validate_args=True,
        ),
    )
    mpi_comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
    namelist = Namelist.from_f90nml(f90nml.read(NAMELIST_FILENAME))
    partitioner = fv3gfs.util.CubedSpherePartitioner(
        fv3gfs.util.TilePartitioner(namelist.layout)
    )
    communicator = fv3gfs.util.CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = fv3gfs.util.SubtileGridSizer.from_tile_params(
        nx_tile=namelist.npx - 1,
        ny_tile=namelist.npy - 1,
        nz=namelist.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=namelist.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=mpi_comm.rank,
    )
    grid_indexing = fv3core.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )
    metric_terms = MetricTerms.from_tile_sizing(
        npx=namelist.npx,
        npy=namelist.npy,
        npz=namelist.npz,
        communicator=communicator,
        backend="numpy",
    )

    # create an initial state from the Jablonowski & Williamson Baroclinic
    # test case perturbation. JRMS2006
    state = baroclinic_init.init_baroclinic_state(
        metric_terms,
        adiabatic=namelist.adiabatic,
        hydrostatic=namelist.hydrostatic,
        moist_phys=namelist.moist_phys,
        comm=communicator,
    )
    stencil_factory = fv3core.StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )

    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=GridData.new_from_metric_terms(metric_terms),
        stencil_factory=stencil_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=namelist.dynamical_core,
        phis=state.phis_quantity,
    )
    do_adiabatic_init = False
    # TODO compute from namelist
    bdt = 225.0

    dycore.step_dynamics(
        state,
        namelist.consv_te,
        do_adiabatic_init,
        bdt,
        namelist.n_split,
    )


def test_call_does_not_allocate_storages():
    call_dycore()


def test_call_does_not_define_stencils():
    pass


def test_call_does_not_access_global_state():
    pass
