import numpy as np
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

from pace.dsl.gt4py_utils import make_storage_from_shape
from pace.dsl.stencil import (
    GridIndexing,
    StencilConfig,
    StencilFactory,
    get_stencils_with_varied_bounds,
)
from pace.dsl.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add_1_stencil(q: FloatField):
    with computation(PARALLEL), interval(...):
        qin = q
        q = qin + 1.0


def add_1_in_region_stencil(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_start

    with computation(PARALLEL), interval(...):
        q_out = q_in
        with horizontal(region[i_start, :]):
            q_out = q_in + 1.0


def setup_data_vars(backend: str):
    shape = (7, 7, 3)
    q = make_storage_from_shape(shape, backend=backend)
    q[:] = 1.0
    q_ref = make_storage_from_shape(shape, backend=backend)
    q_ref[:] = 1.0
    return q, q_ref


def get_stencil_factory(backend: str) -> StencilFactory:
    config = StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    indexing = GridIndexing(
        domain=(12, 12, 79),
        n_halo=3,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )
    return StencilFactory(config=config, grid_indexing=indexing)


def test_get_stencils_with_varied_bounds(backend: str):
    origins = [(2, 2, 0), (1, 1, 0)]
    domains = [(1, 1, 3), (2, 2, 3)]
    factory = get_stencil_factory(backend)
    stencils = get_stencils_with_varied_bounds(
        add_1_stencil, origins, domains, stencil_factory=factory
    )
    assert len(stencils) == len(origins)
    q, q_ref = setup_data_vars(backend=backend)
    stencils[0](q)
    q_ref[2:3, 2:3, :] = 2.0
    np.testing.assert_array_equal(q.data, q_ref.data)
    stencils[1](q)
    q_ref[2:3, 2:3, :] = 3.0
    q_ref[1, 1:3, :] = 2.0
    q_ref[2:3, 1, :] = 2.0
    np.testing.assert_array_equal(q.data, q_ref.data)


def test_get_stencils_with_varied_bounds_and_regions(backend: str):
    factory = get_stencil_factory(backend)
    origins = [(3, 3, 0), (2, 2, 0)]
    domains = [(1, 1, 3), (2, 2, 3)]
    stencils = get_stencils_with_varied_bounds(
        add_1_in_region_stencil,
        origins,
        domains,
        stencil_factory=factory,
    )
    q_orig, q_ref = setup_data_vars(backend=backend)
    stencils[0](q_orig, q_orig)
    q_ref[3, 3] = 2.0
    np.testing.assert_array_equal(q_orig.data, q_ref.data)
    stencils[1](q_orig, q_orig)
    q_ref[3, 2] = 2.0
    q_ref[3, 3] = 3.0
    np.testing.assert_array_equal(q_orig.data, q_ref.data)
