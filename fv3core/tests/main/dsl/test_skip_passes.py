import unittest.mock

from gt4py.gtscript import PARALLEL, computation, interval
from gtc.passes.oir_optimizations.horizontal_execution_merging import (
    HorizontalExecutionMerging,
)
from gtc.passes.oir_pipeline import DefaultPipeline

from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.dsl.typing import FloatField
from pace.util import X_DIM, Y_DIM, Z_DIM


def stencil_definition(a: FloatField):
    with computation(PARALLEL), interval(...):
        a = 0.0


def test_skip_passes_becomes_oir_pipeline():
    config = StencilConfig(backend="numpy")
    grid_indexing = GridIndexing(
        domain=(4, 4, 7),
        n_halo=3,
        south_edge=False,
        north_edge=False,
        west_edge=False,
        east_edge=False,
    )
    factory = StencilFactory(config=config, grid_indexing=grid_indexing)
    with unittest.mock.patch("gt4py.gtscript.stencil") as mock_stencil_builder:
        factory.from_dims_halo(
            stencil_definition,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
    pipeline: DefaultPipeline = mock_stencil_builder.call_args.kwargs.get(
        "oir_pipeline", DefaultPipeline()
    )
    assert HorizontalExecutionMerging not in pipeline.skip
    assert HorizontalExecutionMerging in pipeline.steps
    with unittest.mock.patch("gt4py.gtscript.stencil") as mock_stencil_builder:
        factory.from_dims_halo(
            stencil_definition,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            skip_passes=("HorizontalExecutionMerging",),
        )
    assert "oir_pipeline" in mock_stencil_builder.call_args.kwargs
    pipeline: DefaultPipeline = mock_stencil_builder.call_args.kwargs["oir_pipeline"]
    assert HorizontalExecutionMerging in pipeline.skip
    assert HorizontalExecutionMerging not in pipeline.steps
