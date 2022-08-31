from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

from gt4py import config as gt_config


if TYPE_CHECKING:
    from pace.dsl.stencil_config import CompilationConfig


def determine_rank_is_compiling(rank: int, size: int) -> bool:
    """Determines if a rank needs to be a compiling one

    Args:
        rank (int): current rank
        size (int): size of the communicator

    Returns:
        bool: True if the rank is a compiling one
    """
    return rank < (size / 6)


def block_waiting_for_compilation(comm, compilation_config: CompilationConfig) -> None:
    """block moving on until an ok is received from the compiling rank

    Args:
        comm (MPI.Comm): communicator over which the ok is sent
        stencil_config (CompilationConfig): holding communicator and rank information
    """
    if comm and comm.Get_size() > 1:
        compiling_rank = compilation_config.compiling_equivalent
        _ = comm.recv(source=compiling_rank)


def unblock_waiting_tiles(comm) -> None:
    """sends a message to all the ranks waiting for compilation to finish

    Args:
        comm (MPI.Comm): communicator over which the ok is sent
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    if comm and size > 1:
        for tile in range(1, 6):
            tile_size = size / 6
            message = "compilation finished"
            comm.send(message, dest=tile * tile_size + rank)


def check_cached_path_exists(cache_filepath: str) -> None:
    if not os.path.exists(cache_filepath):
        raise RuntimeError(f"Error: Could not find caches for rank at {cache_filepath}")


def build_cache_path(config: CompilationConfig) -> Tuple[str, str]:
    """generate the GT-Cache path from the config

    Args:
        config (CompilationConfig): stencil-config object at post-init state

    Returns:
        Tuple[str, str]: path and individual rank string
    """
    if config.size == 1:
        target_rank_str = ""
    else:
        if config.use_minimal_caching:
            target_rank_str = f"_{config.compiling_equivalent:06d}"
        else:
            target_rank_str = f"_{config.rank:06d}"

    path = f"{gt_config.cache_settings['root_path']}/.gt_cache{target_rank_str}"
    return path, target_rank_str


def set_distributed_caches(config: CompilationConfig):
    """In Run mode, check required file then point current rank cache to source cache"""

    # Check that we have all the file we need to early out in case
    # of issues.
    from pace.dsl.stencil_config import RunMode

    if config.run_mode == RunMode.Run:
        cache_filepath, target_rank_str = build_cache_path(config)
        check_cached_path_exists(cache_filepath)
        gt_config.cache_settings["dir_name"] = f".gt_cache{target_rank_str}"
        print(
            f"[{config.run_mode}] Rank {config.rank} "
            f"reading cache {gt_config.cache_settings['dir_name']}"
        )
