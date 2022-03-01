#!/usr/bin/env python3
import contextlib
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple

import f90nml

import fv3core
import pace.dsl.stencil
from fv3core._config import DynamicalCoreConfig
from fv3core.initialization.baroclinic import init_baroclinic_state
from fv3core.utils.null_comm import NullComm
from pace.util.grid import DampingCoefficients, GridData, MetricTerms


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


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="gt4py backend to use",
    )

    return parser.parse_args()


def setup_dycore(config, rank, backend) -> Tuple[fv3core.DynamicalCore, List[Any]]:
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )

    mpi_comm = NullComm(
        rank=rank, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
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
    state = init_baroclinic_state(
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


if __name__ == "__main__":
    args = parse_args()
    namelist = f90nml.read(args.data_dir + "/input.nml")
    dycore_config = DynamicalCoreConfig.from_f90nml(namelist)
    for rank in range(dycore_config.layout[0] * dycore_config.layout[1]):
        dycore, dycore_args = setup_dycore(dycore_config, rank, args.backend)
        with no_lagrangian_contributions(dynamical_core=dycore):
            dycore.step_dynamics(*dycore_args)
