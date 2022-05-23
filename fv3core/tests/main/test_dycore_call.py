import contextlib
import os
import unittest.mock
from typing import Any, List, Tuple

import fv3core
import fv3core._config
import fv3core.initialization.baroclinic as baroclinic_init
import pace.dsl.stencil
import pace.stencils.testing
import pace.util
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
from pace.util.null_comm import NullComm


DIR = os.path.abspath(os.path.dirname(__file__))


@contextlib.contextmanager
def no_lagrangian_contributions(dynamical_core: fv3core.DynamicalCore):
    # TODO: lagrangian contributions currently cause an out of bounds iteration
    # when halo updates are disabled. Fix that bug and remove this decorator.
    # Probably requires an update to gt4py (currently v36).
    def do_nothing(*args, **kwargs):
        pass

    original_attributes = {}
    for obj in (
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_pt,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_u,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_v,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_w,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
    ):
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    for (
        obj
    ) in dynamical_core._lagrangian_to_eulerian_obj._mapn_tracer._list_of_remap_objects:
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    try:
        yield
    finally:
        for obj, original in original_attributes.items():
            obj._lagrangian_contributions = original


def setup_dycore() -> Tuple[fv3core.DynamicalCore, List[Any]]:
    backend = "numpy"
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    config = fv3core.DynamicalCoreConfig(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
        ntiles=6,
        nwat=6,
        dt_atmos=225,
        a_imp=1.0,
        beta=0.0,
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.2,
        d2_bg_k2=0.1,
        d4_bg=0.15,
        d_con=1.0,
        d_ext=0.0,
        dddmp=0.5,
        delt_max=0.002,
        do_sat_adj=True,
        do_vort_damp=True,
        fill=True,
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False,
        k_split=1,
        ke_bg=0.0,
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
    )
    mpi_comm = NullComm(
        rank=0, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = pace.util.CubedSpherePartitioner(
        pace.util.TilePartitioner(config.layout)
    )
    communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = pace.util.SubtileGridSizer.from_tile_params(
        nx_tile=config.npx - 1,
        ny_tile=config.npy - 1,
        nz=config.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=config.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )
    quantity_factory = pace.util.QuantityFactory.from_backend(
        sizer=sizer, backend=backend
    )
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
    )

    # create an initial state from the Jablonowski & Williamson Baroclinic
    # test case perturbation. JRMS2006
    state = baroclinic_init.init_baroclinic_state(
        metric_terms,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        comm=communicator,
    )
    stencil_factory = pace.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )

    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=GridData.new_from_metric_terms(metric_terms),
        stencil_factory=stencil_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=config,
        phis=state.phis,
    )
    do_adiabatic_init = False
    # TODO compute from namelist
    bdt = config.dt_atmos

    args = [
        state,
        config.consv_te,
        do_adiabatic_init,
        bdt,
        config.n_split,
    ]
    return dycore, args


def test_call_does_not_allocate_storages():
    dycore, args = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.storage.storage.zeros", new=error_func):
        with unittest.mock.patch("gt4py.storage.storage.empty", new=error_func):
            with no_lagrangian_contributions(dynamical_core=dycore):
                dycore.step_dynamics(*args)


def test_call_does_not_define_stencils():
    dycore, args = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.gtscript.stencil", new=error_func):
        with no_lagrangian_contributions(dynamical_core=dycore):
            dycore.step_dynamics(*args)
