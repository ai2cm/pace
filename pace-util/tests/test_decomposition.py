import os
import unittest.mock
from typing import Tuple

import pytest

from pace.util.decomposition import (
    build_cache_path,
    check_cached_path_exists,
    determine_rank_is_compiling,
)
from pace.util.partitioner import CubedSpherePartitioner, TilePartitioner


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
def test_determine_rank_is_compiling(
    layout: Tuple[int, int], rank: int, is_compiling: bool
):
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    assert determine_rank_is_compiling(rank, partitioner) == is_compiling


def test_determine_rank_is_compiling_large():
    with pytest.raises(RuntimeError):
        partitioner = CubedSpherePartitioner(TilePartitioner((4, 4)))
        determine_rank_is_compiling(0, partitioner)


def test_check_cached_path_exists():
    with pytest.raises(RuntimeError):
        check_cached_path_exists("notarealpath")


def test_check_cached_path_exists_working():
    path = os.getcwd()
    check_cached_path_exists(path)


@pytest.mark.parametrize(
    "use_minimal_caching, compiling_equivalent, rank, size, target_rank_str",
    [
        pytest.param(True, 2, 6, 24, "_000002", id="find_equivalent"),
        pytest.param(False, 2, 6, 24, "_000006", id="find_self"),
        pytest.param(True, 1, 1, 1, "", id="find_nothing"),
        pytest.param(False, 1, 1, 1, "", id="find_nothing again"),
    ],
)
def test_build_cache_path(
    use_minimal_caching: bool,
    compiling_equivalent: int,
    rank: int,
    size: int,
    target_rank_str: str,
):
    compilation_config = unittest.mock.MagicMock(
        use_minimal_caching=use_minimal_caching,
        compiling_equivalent=compiling_equivalent,
        rank=rank,
        size=size,
    )
    _, rank_str = build_cache_path(compilation_config)
    assert rank_str == target_rank_str
