from typing import List, Optional, Tuple

from dace.sdfg import SDFG

from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.util import TilePartitioner


################################################
# Distributed compilation


def determine_compiling_ranks(config: DaceConfig) -> bool:
    is_compiling = False
    rank = config.my_rank
    size = config.rank_size

    if int(size / 6) == 0:
        is_compiling = True
    elif rank % int(size / 6) == rank:
        is_compiling = True

    return is_compiling


def unblock_waiting_tiles(comm, sdfg_path: str) -> None:
    if comm and comm.Get_size() > 1:
        for tile in range(1, 6):
            tilesize = comm.Get_size() / 6
            comm.send(sdfg_path, dest=tile * tilesize + comm.Get_rank())


def get_target_rank(rank: int, partitioner: TilePartitioner):
    """From my rank & the current partitioner we determine which
    rank we should read from.
    For all layout >= 3,3 this presumes build has been done on a
    3,3 layout."""
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
    else:
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


def build_info_filepath() -> str:
    return "build_info.txt"


def write_build_info(
    sdfg: SDFG, layout: Tuple[int], resolution_per_tile: List[int], backend: str
):
    """Write down all relevant information on the build to identify
    it at load time."""
    # Dev NOTE: we should be able to leverage sdfg.make_key to get a hash or
    # even go to a complete hash base system and read the data from the SDFG itself
    import os

    path_to_sdfg_dir = os.path.abspath(sdfg.build_folder)
    with open(f"{path_to_sdfg_dir}/{build_info_filepath()}", "w") as build_info_read:
        build_info_read.write("#Schema: Backend Layout Resolution per tile\n")
        build_info_read.write(f"{backend}\n")
        build_info_read.write(f"{str(layout)}\n")
        build_info_read.write(f"{str(resolution_per_tile)}\n")


################################################

################################################
# SDFG load (both .sdfg file and build directory containing .so)


def get_sdfg_path(
    daceprog_name: str, config: DaceConfig, sdfg_file_path: Optional[str] = None
) -> Optional[str]:
    """Build an SDFG path from the qualified program name or it's direct path to .sdfg

    Args:
        program_name: qualified name in the form module_qualname if module is not locals
        sdfg_file_path: absolute path to a .sdfg file
    """
    import os

    # TODO: check DaceConfig for cache.strategy == name
    # Guarding against bad usage of this function
    if config.get_orchestrate() != DaCeOrchestration.Run:
        return None

    # Case of a .sdfg file given by the user to be compiled
    if sdfg_file_path is not None:
        if not os.path.isfile(sdfg_file_path):
            raise RuntimeError(
                f"SDFG filepath {sdfg_file_path} cannot be found or is not a file"
            )
        return sdfg_file_path

    # Case of loading a precompiled .so - lookup using GT_CACHE
    from gt4py import config as gt_config

    if config.rank_size > 1:
        rank = config.my_rank
        rank_str = f"_{config.target_rank:06d}"
    else:
        rank = 0
        rank_str = f"_{rank:06d}"

    sdfg_dir_path = (
        f"{gt_config.cache_settings['root_path']}"
        f"/.gt_cache{rank_str}/dacecache/{daceprog_name}"
    )
    if not os.path.isdir(sdfg_dir_path):
        raise RuntimeError(f"Precompiled SDFG is missing at {sdfg_dir_path}")

    # Check layout in build time matches layout now
    import ast

    with open(f"{sdfg_dir_path}/{build_info_filepath()}") as build_info_file:
        # Jump over schema comment
        build_info_file.readline()
        # Read in
        build_backend = build_info_file.readline().rstrip()
        if config.get_backend() != build_backend:
            raise RuntimeError(
                f"SDFG build for {build_backend}, {config._backend} has been asked"
            )
        # Check layout
        build_layout = ast.literal_eval(build_info_file.readline())
        can_read = True
        if config.layout == (1, 1) and config.layout != build_layout:
            can_read = False
        elif config.layout == (2, 2) and config.layout != build_layout:
            can_read = False
        elif (
            build_layout != (1, 1) and build_layout != (2, 2) and build_layout != (3, 3)
        ):
            can_read = False
        if not can_read:
            raise RuntimeError(
                f"SDFG build for layout {build_layout}, "
                f"cannot be run with current layout {config.layout}"
            )
        # Check resolution per tile
        build_resolution = ast.literal_eval(build_info_file.readline())
        if config.tile_resolution != build_resolution:
            raise RuntimeError(
                f"SDFG build for resolution {build_resolution}, "
                f"cannot be run with current resolution {config.tile_resolution}"
            )

    print(f"[DaCe Config] Rank {rank} loading SDFG {sdfg_dir_path}")

    return sdfg_dir_path


def set_distributed_caches(config: "DaceConfig"):
    """In Run mode, check required file then point current rank cache to source cache"""

    # Execute specific initialization per orchestration state
    orchestration_mode = config.get_orchestrate()

    # Check that we have all the file we need to early out in case
    # of issues.
    if orchestration_mode == DaCeOrchestration.Run:
        import os

        from gt4py import config as gt_config

        # Check our cache exist
        if config.rank_size > 1:
            rank = config.my_rank
            target_rank_str = f"_{config.target_rank:06d}"
        else:
            rank = 0
            target_rank_str = f"_{rank:06d}"
        cache_filepath = (
            f"{gt_config.cache_settings['root_path']}/.gt_cache{target_rank_str}"
        )
        if not os.path.exists(cache_filepath):
            raise RuntimeError(
                f"{orchestration_mode} error: Could not find caches for rank "
                f"{rank} at {cache_filepath}"
            )

        # All, good set this rank cache to the source cache
        gt_config.cache_settings["dir_name"] = f".gt_cache{target_rank_str}"
        print(
            f"[{orchestration_mode}] Rank {rank} "
            f"reading cache {gt_config.cache_settings['dir_name']}"
        )
