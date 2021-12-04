import os
from typing import Any, List, Tuple

import fv3core.initialization.baroclinic as baroclinic_init
from fv3core.grid import MetricTerms
from fv3core.utils.grid import DampingCoefficients, GridData


# Dev note: the GTC toolchain fails if xarray is imported after gt4py
# pace.util imports xarray if it's available in the env.
# fv3core imports gt4py.
# To avoid future conflict creeping back we make util imported prior to
# fv3core. isort turned off to keep it that way.
# isort: off
import pace.util
from fv3core.utils.null_comm import NullComm

# isort: on

import contextlib
import unittest.mock

import f90nml

import fv3core
import fv3core._config
import fv3core.testing
from fv3core._config import Namelist


DIR = os.path.abspath(os.path.dirname(__file__))
NAMELIST_FILENAME = os.path.join(DIR, "c12_namelist.nml")


@contextlib.contextmanager
def no_lagrangian_contributions(dynamical_core: fv3core.DynamicalCore):
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
    stencil_config = fv3core.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    mpi_comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
    namelist = Namelist.from_f90nml(f90nml.read(NAMELIST_FILENAME))
    partitioner = pace.util.CubedSpherePartitioner(
        pace.util.TilePartitioner(namelist.layout)
    )
    communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = pace.util.SubtileGridSizer.from_tile_params(
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
        backend=backend,
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

    # TODO: lagrangian contributions currently cause an out of bounds iteration
    # when halo updates are disabled. Fix that bug and remove this decorator.
    args = [
        state,
        namelist.consv_te,
        do_adiabatic_init,
        bdt,
        namelist.n_split,
    ]
    return dycore, args


def test_call_does_not_access_global_state():
    dycore, args = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    mock_grid = unittest.mock.MagicMock()
    with unittest.mock.patch("fv3core.utils.global_config.get_backend", new=error_func):
        with unittest.mock.patch(
            "fv3core.utils.global_config.is_gpu_backend", new=error_func
        ):
            with unittest.mock.patch("fv3core._config.set_grid", new=error_func):
                with unittest.mock.patch("fv3core._config.grid", new=mock_grid):
                    with no_lagrangian_contributions(dynamical_core=dycore):
                        dycore.step_dynamics(*args)
    mock_grid.assert_not_called()


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
