import os
from typing import List, Optional, Tuple

import gt4py.config as config
from gt4py import config as gt_config

from pace.dsl.dace.dace_config import DaceConfig
from pace.dsl.stencil import CompilationConfig, RunMode, StencilConfig
from pace.util import TilePartitioner
from pace.util.partitioner import CubedSpherePartitioner


################################################
# Distributed compilation


def compiling_equivalent(rank: int, partitioner: TilePartitioner):
    """From my rank & the current partitioner we determine which
    rank we should read from"""
    if partitioner.layout == (1, 1):
        return 0
    if partitioner.layout == (2, 2):
        if partitioner.tile.on_tile_bottom(rank):
            if partitioner.tile.on_tile_left(rank):
                return 0  # "00"
            if partitioner.tile.on_tile_right(rank):
                return 1  # "10"
        if partitioner.tile.on_tile_top(rank):
            if partitioner.tile.on_tile_left(rank):
                return 2  # "01"
            if partitioner.tile.on_tile_right(rank):
                return 3  # "11"
    if partitioner.layout == (3, 3):
        if partitioner.tile.on_tile_bottom(rank):
            if partitioner.tile.on_tile_left(rank):
                return 0  # "00"
            if partitioner.tile.on_tile_right(rank):
                return 2  # "20"
            else:
                return 1  # "10"
        if partitioner.tile.on_tile_top(rank):
            if partitioner.tile.on_tile_left(rank):
                return 6  # "02"
            if partitioner.tile.on_tile_right(rank):
                return 8  # "22"
            else:
                return 7  # "12"
        else:
            if partitioner.tile.on_tile_left(rank):
                return 3  # "01"
            if partitioner.tile.on_tile_right(rank):
                return 5  # "21"
            else:
                return 4  # "11"
    else:
        raise RuntimeError(
            "Can't compile with a layout larger than 3x3 with minimal caching on"
        )


def determine_rank_is_compiling(rank: int, partitioner: CubedSpherePartitioner) -> bool:
    top_tile_equivalent = compiling_equivalent(rank, partitioner)
    return rank == top_tile_equivalent


def unblock_waiting_tiles(comm) -> None:
    """sends a message to all the ranks waiting for compilation to finish

    Args:
        stencil_config (CompilationConfig): configuration that holds the communicator and the rank information
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    if comm and size > 1:
        for tile in range(1, 6):
            tile_size = size / 6
            message = "compilation finished"
            comm.send(message, dest=tile * tile_size + rank)


def set_distributed_caches(config: CompilationConfig):
    """In Run mode, check required file then point current rank cache to source cache"""

    # Check that we have all the file we need to early out in case
    # of issues.
    if config.run_mode == RunMode.Run:
        rank = config.rank
        if config.size == 1:
            target_rank_str = ""
        else:
            if config.use_minimal_caching:
                target_rank_str = f"_{config.compiling_equivalent:06d}"
            else:
                target_rank_str = f"_{config.rank:06d}"

        cache_filepath = (
            f"{gt_config.cache_settings['root_path']}/.gt_cache{target_rank_str}"
        )
        if not os.path.exists(cache_filepath):
            raise RuntimeError(
                f"{config.run_mode} error: Could not find caches for rank "
                f"{rank} at {cache_filepath}"
            )

        # All, good set this rank cache to the source cache
        gt_config.cache_settings["dir_name"] = f".gt_cache{target_rank_str}"
        print(
            f"[{config.run_mode}] Rank {rank} "
            f"reading cache {gt_config.cache_settings['dir_name']}"
        )


def build_info_filepath() -> str:
    return "build_info.txt"


def write_build_info(layout: Tuple[int], resolution_per_tile: List[int], backend: str):
    """Write down all relevant information on the build to identify
    it at load time."""

    path = config.cache_settings["root_path"]
    with open(f"{path}/{build_info_filepath()}", "w") as build_info_read:
        build_info_read.write("#Schema: Backend Layout\n")
        build_info_read.write(f"{backend}\n")
        build_info_read.write(f"{str(layout)}\n")


# def compiling_equivalent(rank: int, partitioner: CubedSpherePartitioner) -> int:
#     if partitioner.tile.on_tile_bottom(rank):
#         if partitioner.tile.on_tile_left(rank):
#             return get_first_matching_rank(partitioner, "00")
#         if partitioner.tile.on_tile_right(rank):
#             return get_first_matching_rank(partitioner, "20")
#         else:
#             return get_first_matching_rank(partitioner, "10")
#     if partitioner.tile.on_tile_top(rank):
#         if partitioner.tile.on_tile_left(rank):
#             return get_first_matching_rank(partitioner, "02")
#         if partitioner.tile.on_tile_right(rank):
#             return get_first_matching_rank(partitioner, "22")
#         else:
#             return get_first_matching_rank(partitioner, "12")
#     else:
#         if partitioner.tile.on_tile_left(rank):
#             return get_first_matching_rank(partitioner, "01")
#         if partitioner.tile.on_tile_right(rank):
#             return get_first_matching_rank(partitioner, "21")
#         else:
#             return get_first_matching_rank(partitioner, "11")

# def get_first_matching_rank(
#     partitioner: CubedSpherePartitioner, decomposition_string: str
# ) -> int:
#     tile_size = partitioner.total_ranks / 6
#     for rank in range(partitioner.total_ranks):
#         if (
#             decomposition_string == "00"
#             and partitioner.tile.on_tile_bottom(rank)
#             and partitioner.tile.on_tile_left(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "10"
#             and partitioner.tile.on_tile_bottom(rank)
#             and not partitioner.tile.on_tile_left(rank)
#             and not partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "20"
#             and partitioner.tile.on_tile_bottom(rank)
#             and partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "01"
#             and not partitioner.tile.on_tile_bottom(rank)
#             and not partitioner.tile.on_tile_top(rank)
#             and partitioner.tile.on_tile_left(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "11"
#             and not partitioner.tile.on_tile_bottom(rank)
#             and not partitioner.tile.on_tile_top(rank)
#             and not partitioner.tile.on_tile_left(rank)
#             and not partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "21"
#             and not partitioner.tile.on_tile_bottom(rank)
#             and not partitioner.tile.on_tile_top(rank)
#             and partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "02"
#             and partitioner.tile.on_tile_top(rank)
#             and partitioner.tile.on_tile_left(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "12"
#             and partitioner.tile.on_tile_top(rank)
#             and not partitioner.tile.on_tile_left(rank)
#             and not partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size
#         if (
#             decomposition_string == "22"
#             and partitioner.tile.on_tile_top(rank)
#             and partitioner.tile.on_tile_right(rank)
#         ):
#             return rank % tile_size

#     raise RuntimeError("Tiling seems to be broken")
