from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.storage import empty, ones

import pace.dsl
from pace.dsl.stencil import CompilationConfig, GridIndexing


def _make_storage(
    func,
    grid_indexing,
    stencil_config: pace.dsl.StencilConfig,
    *,
    dtype=float,
    aligned_index=(0, 0, 0),
):
    return func(
        backend=stencil_config.compilation_config.backend,
        shape=grid_indexing.domain,
        dtype=dtype,
        aligned_index=aligned_index,
    )


def test_timing_collector():
    grid_indexing = GridIndexing(
        domain=(5, 5, 5),
        n_halo=2,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )
    stencil_config = pace.dsl.StencilConfig(
        compilation_config=CompilationConfig(backend="numpy", rebuild=True)
    )

    stencil_factory = pace.dsl.StencilFactory(stencil_config, grid_indexing)

    def func(inp: Field[float], out: Field[float]):
        with computation(PARALLEL), interval(...):
            out = inp

    test = stencil_factory.from_origin_domain(
        func, (0, 0, 0), domain=grid_indexing.domain
    )

    build_report = stencil_factory.build_report(key="parse_time")
    assert "func" in build_report

    inp = _make_storage(ones, grid_indexing, stencil_config, dtype=float)
    out = _make_storage(empty, grid_indexing, stencil_config, dtype=float)

    test(inp, out)
    exec_report = stencil_factory.exec_report()
    assert "func" in exec_report
