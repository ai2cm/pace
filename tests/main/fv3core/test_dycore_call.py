import os
import unittest.mock
from dataclasses import fields
from datetime import timedelta
from typing import Tuple

import pace.dsl.stencil
import pace.fv3core.initialization.baroclinic as baroclinic_init
import pace.stencils.testing
import pace.util
from pace import fv3core
from pace.dsl.dace.dace_config import DaceConfig
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.stencils.testing import assert_same_temporaries, copy_temporaries
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
from pace.util.null_comm import NullComm


DIR = os.path.abspath(os.path.dirname(__file__))


def setup_dycore() -> Tuple[
    fv3core.DynamicalCore, fv3core.DycoreState, pace.util.Timer
]:
    backend = "numpy"
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
    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = pace.dsl.stencil.StencilConfig(
        compilation_config=pace.dsl.stencil.CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=dace_config,
    )
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
    grid_data = GridData.new_from_metric_terms(metric_terms)

    # create an initial state from the Jablonowski & Williamson Baroclinic
    # test case perturbation. JRMS2006
    state = baroclinic_init.init_baroclinic_state(
        grid_data,
        quantity_factory=quantity_factory,
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
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=config,
        timestep=timedelta(seconds=config.dt_atmos),
        phis=state.phis,
        state=state,
    )

    return dycore, state, pace.util.NullTimer()


def copy_state(state1: DycoreState, state2: DycoreState):
    # copy all attributes of state1 to state2
    for attr_name in dir(state1):
        for _field in fields(type(state1)):
            if issubclass(_field.type, pace.util.Quantity):
                attr = getattr(state1, attr_name)
                if isinstance(attr, pace.util.Quantity):
                    getattr(state2, attr_name).data[:] = attr.data


def test_temporaries_are_deterministic():
    """
    This is a precursor test to the next one, ensuring that two
    identically-initialized dycores called on identically-initialized
    states produce identical temporaries.

    This will fail if there is non-determinism in the initialization,
    for example from using `empty` instead of `zeros` to initialize data.
    """
    dycore1, state1, timer1 = setup_dycore()
    dycore2, state2, timer2 = setup_dycore()

    dycore1.step_dynamics(state1, timer1)
    first_temporaries = copy_temporaries(dycore1, max_depth=10)
    assert len(first_temporaries) > 0
    dycore2.step_dynamics(state2, timer2)
    second_temporaries = copy_temporaries(dycore2, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)


def test_call_on_same_state_same_dycore_produces_same_temporaries():
    """
    Assuming the precursor test passes, this test indicates whether
    the dycore retains and re-uses internal state on subsequent calls.
    If it does not, then subsequent calls on identical input should
    produce identical results.
    """
    dycore, state_1, timer_1 = setup_dycore()
    _, state_2, timer_2 = setup_dycore()

    # state_1 and state_2 are identical, if the dycore is stateless then they
    # should produce identical dycore final states when used to call
    dycore.step_dynamics(state_1, timer_1)
    first_temporaries = copy_temporaries(dycore, max_depth=10)
    assert len(first_temporaries) > 0
    # TODO: The orchestrated code pushed us to make the dycore stateful for halo
    # exchange, so we must copy into state_1 instead of using state_2.
    # We should call with state_2 directly when this is fixed.
    copy_state(state_2, state_1)
    dycore.step_dynamics(state_1, timer_2)
    second_temporaries = copy_temporaries(dycore, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)


def test_call_does_not_allocate_storages():
    dycore, state, timer = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.storage.storage.zeros", new=error_func):
        with unittest.mock.patch("gt4py.storage.storage.empty", new=error_func):
            dycore.step_dynamics(state, timer)


def test_call_does_not_define_stencils():
    dycore, state, timer = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.gtscript.stencil", new=error_func):
        dycore.step_dynamics(state, timer)
