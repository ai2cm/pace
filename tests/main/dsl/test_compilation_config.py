import unittest.mock
from math import sqrt

import pytest

from pace.dsl.stencil import CompilationConfig, RunMode
from pace.util.communicator import CubedSphereCommunicator
from pace.util.partitioner import CubedSpherePartitioner, TilePartitioner


def test_safety_checks():
    with pytest.raises(RuntimeError):
        CompilationConfig(backend="numpy", device_sync=True)
    with pytest.raises(RuntimeError):
        CompilationConfig(backend="gt:cpu_ifirst", device_sync=True)


@pytest.mark.parametrize(
    "size, use_minimal_caching, run_mode",
    [
        pytest.param(54, True, RunMode.Run, id="3x3 layout Run minimal"),
        pytest.param(96, False, RunMode.BuildAndRun, id="4x4 layout BnR normal"),
        pytest.param(96, True, RunMode.Run, id="4x4 layout Run minimal"),
    ],
)
def test_check_communicator_valid(
    size: int, use_minimal_caching: bool, run_mode: RunMode
):
    partitioner = CubedSpherePartitioner(
        TilePartitioner((sqrt(size / 6), (sqrt(size / 6))))
    )
    comm = unittest.mock.MagicMock()
    comm.Get_size.return_value = size
    cubed_sphere_comm = CubedSphereCommunicator(comm, partitioner)
    config = CompilationConfig(
        run_mode=run_mode, use_minimal_caching=use_minimal_caching
    )
    config.check_communicator(cubed_sphere_comm)


@pytest.mark.parametrize(
    "nx, ny, use_minimal_caching, run_mode",
    [
        pytest.param(2, 3, False, RunMode.BuildAndRun, id="2x3 layout BnR normal"),
    ],
)
def test_check_communicator_invalid(
    nx: int, ny: int, use_minimal_caching: bool, run_mode: RunMode
):
    partitioner = CubedSpherePartitioner(TilePartitioner((nx, ny)))
    comm = unittest.mock.MagicMock()
    comm.Get_size.return_value = nx * ny * 6
    cubed_sphere_comm = CubedSphereCommunicator(comm, partitioner)
    config = CompilationConfig(
        run_mode=run_mode, use_minimal_caching=use_minimal_caching
    )
    with pytest.raises(RuntimeError):
        config.check_communicator(cubed_sphere_comm)


def test_get_decomposition_info_from_no_comm():
    config = CompilationConfig()
    (
        computed_rank,
        computed_size,
        computed_equivalent,
        computed_is_compiling,
    ) = config.get_decomposition_info_from_comm(None)
    assert computed_rank == 1
    assert computed_size == 1
    assert computed_equivalent == 1
    assert computed_is_compiling is True


@pytest.mark.parametrize(
    "rank, size, is_compiling, equivalent",
    [
        pytest.param(0, 6, True, 0, id="1x1 layout - 0"),
        pytest.param(1, 6, False, 0, id="1x1 layout - 1"),
        pytest.param(2, 24, True, 2, id="2x2 layout - 2"),
        pytest.param(4, 24, False, 0, id="2x2 layout - 2"),
    ],
)
def test_get_decomposition_info_from_comm(
    rank: int, size: int, is_compiling: bool, equivalent: int
):
    partitioner = CubedSpherePartitioner(
        TilePartitioner((sqrt(size / 6), sqrt(size / 6)))
    )
    comm = unittest.mock.MagicMock()
    comm.Get_rank.return_value = rank
    comm.Get_size.return_value = size
    cubed_sphere_comm = CubedSphereCommunicator(comm, partitioner)
    config = CompilationConfig(use_minimal_caching=True, run_mode=RunMode.Run)
    (
        computed_rank,
        computed_size,
        computed_equivalent,
        computed_is_compiling,
    ) = config.get_decomposition_info_from_comm(cubed_sphere_comm)
    assert rank == computed_rank
    assert size == computed_size
    assert equivalent == computed_equivalent
    assert is_compiling == computed_is_compiling


@pytest.mark.parametrize(
    "rank, size, minimal_caching, run_mode, equivalent",
    [
        pytest.param(0, 6, True, RunMode.Run, 0, id="1x1 layout - 0 - R"),
        pytest.param(1, 6, False, RunMode.Run, 0, id="1x1 layout - 1 - R"),
        pytest.param(2, 24, True, RunMode.Run, 2, id="2x2 layout - 2 - R"),
        pytest.param(4, 24, False, RunMode.Run, 0, id="2x2 layout - 4 - R"),
        pytest.param(5, 54, True, RunMode.Run, 5, id="3x3 layout - 5 - R"),
        pytest.param(28, 54, False, RunMode.Run, 1, id="3x3 layout - 28 - R"),
        pytest.param(10, 96, False, RunMode.Run, 4, id="4x4 layout - 10 - R"),
        pytest.param(20, 96, False, RunMode.Run, 3, id="4x4 layout - 20 - R"),
        pytest.param(
            10, 96, False, RunMode.BuildAndRun, 10, id="4x4 layout - 10 - BnR"
        ),
        pytest.param(20, 96, False, RunMode.BuildAndRun, 4, id="4x4 layout - 20 - BnR"),
    ],
)
def test_determine_compiling_equivalent(
    rank, size, minimal_caching, run_mode, equivalent
):
    config = CompilationConfig(use_minimal_caching=minimal_caching, run_mode=run_mode)
    partitioner = CubedSpherePartitioner(
        TilePartitioner((sqrt(size / 6), sqrt(size / 6)))
    )
    comm = unittest.mock.MagicMock()
    comm.Get_rank.return_value = rank
    comm.Get_size.return_value = size
    cubed_sphere_comm = CubedSphereCommunicator(comm, partitioner)
    assert (
        config.determine_compiling_equivalent(rank, cubed_sphere_comm.partitioner)
        == equivalent
    )


def test_as_dict():
    config = CompilationConfig()
    asdict = config.as_dict()
    assert asdict["backend"] == "numpy"
    assert asdict["rebuild"] is True
    assert asdict["validate_args"] is True
    assert asdict["format_source"] is False
    assert asdict["device_sync"] is False
    assert asdict["run_mode"] == "BuildAndRun"
    assert asdict["use_minimal_caching"] is False
    assert len(asdict) == 7


def test_from_dict():
    specification_dict = {}
    config = CompilationConfig.from_dict(specification_dict)
    assert config.backend == "numpy"
    assert config.rebuild is False
    assert config.validate_args is True
    assert config.format_source is False
    assert config.device_sync is False
    assert config.run_mode == RunMode.BuildAndRun
    assert config.use_minimal_caching is False

    specification_dict["backend"] = "gt:gpu"
    config = CompilationConfig.from_dict(specification_dict)
    assert config.backend == "gt:gpu"

    specification_dict["rebuild"] = True
    config = CompilationConfig.from_dict(specification_dict)
    assert config.rebuild is True

    specification_dict["validate_args"] = False
    config = CompilationConfig.from_dict(specification_dict)
    assert config.validate_args is False

    specification_dict["format_source"] = True
    config = CompilationConfig.from_dict(specification_dict)
    assert config.format_source is True

    specification_dict["device_sync"] = True
    config = CompilationConfig.from_dict(specification_dict)
    assert config.device_sync is True

    specification_dict["run_mode"] = "Build"
    config = CompilationConfig.from_dict(specification_dict)
    assert config.run_mode == RunMode.Build

    specification_dict["use_minimal_caching"] = True
    specification_dict["run_mode"] = "Run"
    config = CompilationConfig.from_dict(specification_dict)
    assert config.use_minimal_caching is True
