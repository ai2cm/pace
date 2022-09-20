import copy
import math
import os
import shutil
from argparse import ArgumentParser

import gt4py
import yaml
from mpi4py import MPI

from pace.driver.comm import NullCommConfig
from pace.driver.run import Driver, DriverConfig


def parse_args():
    usage = "usage: python %(prog)s config_file"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "config_file",
        type=str,
        action="store",
        help="which config file to use",
    )
    parser.add_argument("target_dir", type=str, action="store", help="")
    return parser.parse_args()


def get_iterations_from_config(driver_config: DriverConfig):
    sub_tiles = (
        driver_config.dycore_config.layout[0] * driver_config.dycore_config.layout[1]
    )
    iterations = math.ceil(sub_tiles / size)
    return iterations, sub_tiles


def setup_driver_objects(global_rank: int, size: int, driver_config: DriverConfig):
    iterations, sub_tiles = get_iterations_from_config(driver_config)
    gt4py.config.cache_settings["root_path"] = os.environ.get("GT_CACHE_DIR_NAME", ".")
    for iteration in range(iterations):
        top_tile_rank = global_rank + size * iteration
        if top_tile_rank < sub_tiles:
            my_driver_config = copy.deepcopy(driver_config)
            assert isinstance(my_driver_config.comm_config.config, NullCommConfig)
            my_driver_config.comm_config.config.rank = top_tile_rank
            gt4py.config.cache_settings["dir_name"] = os.environ.get(
                "GT_CACHE_ROOT", f".gt_cache_{top_tile_rank:06}"
            )
            driver = Driver(
                config=my_driver_config,
            )
    if comm is not None:
        comm.Barrier()


def copy_caches_to_target_location(
    driver_config: DriverConfig, global_rank: int, size: int, target_dir: str
):
    iterations, sub_tiles = get_iterations_from_config(driver_config)

    root_path = gt4py.config.cache_settings["root_path"]
    for iter in range(iterations):
        top_tile_rank = global_rank + size * iter
        if top_tile_rank < sub_tiles:
            for tile in range(6):
                shutil.copytree(
                    f"{root_path}/.gt_cache_{top_tile_rank:06}",
                    f"{target_dir}/.gt_cache" f"_{(top_tile_rank + tile*sub_tiles):06}",
                    dirs_exist_ok=True,
                )
                print(
                    f"rank {global_rank} copied for "
                    f"target rank {(top_tile_rank + tile*sub_tiles)}"
                )


if __name__ == "__main__":
    args = parse_args()
    if MPI is not None:
        comm = MPI.COMM_WORLD
        global_rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        global_rank = 0
        size = 1
    with open(args.config_file, "r") as f:
        driver_config = DriverConfig.from_dict(yaml.safe_load(f))
    setup_driver_objects(global_rank, size, driver_config)
    copy_caches_to_target_location(driver_config, global_rank, size, args.target_dir)

    if comm is not None:
        comm.Barrier()
    print(f"rank {global_rank} is finished, exiting")
    if global_rank == 0:
        print("SUCCESS")
