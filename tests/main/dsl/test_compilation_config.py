import unittest.mock as mock
from typing import Tuple

import pytest

from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import CompilationConfig, RunMode, StencilConfig
from pace.util.decomposition import determine_rank_is_compiling
from pace.util.partitioner import CubedSpherePartitioner, Partitioner, TilePartitioner


@pytest.mark.parametrize(
    "layout, rank, is_compiling",
    [
        pytest.param((1, 1), 0, True, id="1x1 layout"),
        pytest.param((1, 1), 2, False, id="1x1 layout"),
        pytest.param((2, 2), 1, True, id="2x2 layout"),
        pytest.param((2, 2), 5, False, id="2x2 layout"),
        pytest.param((3, 3), 8, True, id="3x3 layout"),
        pytest.param((3, 3), 25, False, id="3x3 layout"),
    ],
)
def test_compiling_ranks(layout: Tuple[int, int], rank: int, is_compiling: bool):
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    assert determine_rank_is_compiling(rank, partitioner) == is_compiling


def test_assert_on_large_layout():
    with pytest.raises(RuntimeError):
        partitioner = CubedSpherePartitioner(TilePartitioner((4, 4)))
        determine_rank_is_compiling(0, partitioner)


def tests_configurations():
    config = CompilationConfig()  # use_minimal_caching=True, run_mode=RunMode.Run)


# @pytest.mark.parametrize("validate_args", [True, False])
# @pytest.mark.parametrize("device_sync", [True, False])
# @pytest.mark.parametrize("rebuild", [True, False])
# @pytest.mark.parametrize("format_source", [True, False])
# def test_same_config_equal(
#     backend: str,
#     rebuild: bool,
#     validate_args: bool,
#     format_source: bool,
#     device_sync: bool,
# ):
#     dace_config = DaceConfig(None, backend, DaCeOrchestration.Python,)
#     config = StencilConfig(
#         compilation_config=CompilationConfig(
#             backend=backend,
#             rebuild=rebuild,
#             validate_args=validate_args,
#             format_source=format_source,
#             device_sync=device_sync,
#         ),
#         dace_config=dace_config,
#     )
#     assert config == config

#     same_config = StencilConfig(
#         compilation_config=CompilationConfig(
#             backend=backend,
#             rebuild=rebuild,
#             validate_args=validate_args,
#             format_source=format_source,
#             device_sync=device_sync,
#         ),
#     )
#     assert config == same_config
