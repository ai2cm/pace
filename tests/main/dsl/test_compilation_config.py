import unittest.mock as mock

import pytest

from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import CompilationConfig, StencilConfig
from pace.util.decomposition import determine_compiling_ranks


def test_compiling_ranks():
    part = mock.MagicMock(total_ranks=6)
    comm = mock.MagicMock(rank=0, partitioner=part)
    config = CompilationConfig(communicator=comm, use_minimal_caching=True)

    assert determine_compiling_ranks(config) == True

    part = mock.MagicMock(total_ranks=24)
    comm = mock.MagicMock(rank=5, partitioner=part)
    config = CompilationConfig(communicator=comm, use_minimal_caching=True)
    assert determine_compiling_ranks(config) == False


def tests_configurations():
    config = CompilationConfig(use_minimal_caching=True)


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
