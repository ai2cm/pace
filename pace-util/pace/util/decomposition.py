from typing import Optional
import pace.util.global_config as global_config
from pace.util.partitioner import CubedSpherePartitioner
import yaml
import os


def top_tile_rank_from_decomposition_string(
    decomposition_string: str, partitioner: CubedSpherePartitioner
) -> Optional[int]:
    tilesize = partitioner.total_ranks // 6
    if tilesize == 1:
        return 0
    for rank in range(partitioner.total_ranks):
        if (
            decomposition_string == "00"
            and partitioner.tile.on_tile_bottom(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "10"
            and partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "20"
            and partitioner.tile.on_tile_bottom(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "01"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "11"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "21"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "02"
            and partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "12"
            and partitioner.tile.on_tile_top(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            decomposition_string == "22"
            and partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize

    return None


def top_tile_rank_to_decomposition_string(rank: int) -> str:
    partitioner = global_config.get_partitioner()
    if partitioner.tile.on_tile_bottom(rank):
        if partitioner.tile.on_tile_left(rank):
            return "00"
        if partitioner.tile.on_tile_right(rank):
            return "20"
        else:
            return "10"
    if partitioner.tile.on_tile_top(rank):
        if partitioner.tile.on_tile_left(rank):
            return "02"
        if partitioner.tile.on_tile_right(rank):
            return "22"
        else:
            return "12"
    else:
        if partitioner.tile.on_tile_left(rank):
            return "01"
        if partitioner.tile.on_tile_right(rank):
            return "21"
        else:
            return "11"


def read_target_rank(rank, filename=None):
    partitioner = global_config.get_partitioner()
    top_tile_rank = top_tile_equivalent(rank, partitioner.total_ranks)
    with open(filename) as decomposition:
        parsed_file = yaml.safe_load(decomposition)
        return int(parsed_file[top_tile_rank_to_decomposition_string(top_tile_rank)])


def top_tile_equivalent(rank, size):
    tilesize = size / 6
    return rank % tilesize


def write_decomposition():
    from gt4py import config as gt_config

    partitioner = global_config.get_partitioner()
    path = f"{gt_config.cache_settings['root_path']}/.layout/"
    config_path = path + "decomposition.yml"
    os.makedirs(path, exist_ok=True)
    decomposition = {}
    decomposition["layout"] = partitioner.layout
    for string in ["00", "10", "20", "01", "11", "21", "02", "12", "22"]:
        target_rank = top_tile_rank_from_decomposition_string(string, partitioner)
        if target_rank is not None:
            decomposition.setdefault(string, int(target_rank))

    with open(config_path, "w") as outfile:
        yaml.dump(decomposition, outfile)


def set_distributed_caches(rank, size):
    """In Run mode, check required file then point current rank cache to source cache"""

    # Check that we have all the file we need to early out in case
    # of issues.
    from gt4py import config as gt_config
    import os

    # Check layout
    layout_filepath = (
        f"{gt_config.cache_settings['root_path']}/.layout/decomposition.yml"
    )
    if not os.path.exists(layout_filepath):
        raise RuntimeError(f"error: Could not find layout at {layout_filepath}")

    # Check our cache exist
    if size > 1:
        rank_str = f"_{read_target_rank(rank, layout_filepath):06d}"
    else:
        rank_str = ""
    cache_filepath = f"{gt_config.cache_settings['root_path']}/.gt_cache{rank_str}"
    if not os.path.exists(cache_filepath):
        raise RuntimeError(
            f"error: Could not find caches for rank {rank} at {cache_filepath}"
        )

    # All, good set this rank cache to the source cache
    gt_config.cache_settings["dir_name"] = f".gt_cache{rank_str}"
    print(f"Rank {rank} reading cache {gt_config.cache_settings['dir_name']}")
