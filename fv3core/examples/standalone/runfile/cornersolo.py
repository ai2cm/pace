#!/usr/bin/env python3
import math
from typing import Any, Dict, List, Tuple
import gt4py.storage as gt_storage
import f90nml
from gt4py.gtscript import PARALLEL, computation, horizontal, interval
import numpy as np
from pace.dsl.typing import FloatField
from pace.stencils import corners
import serialbox
import yaml
from mpi4py import MPI

# NOTE: we need to import dsl.stencil prior to
# pace.util, otherwise xarray precedes gt4py, causing
# very strange errors on some systems (e.g. daint)
import pace.dsl.stencil
import pace.util as util
import pace.dsl.gt4py_utils as utils
from fv3core._config import DynamicalCoreConfig
from pace.dsl import StencilFactory
from pace.dsl.dace.orchestrate import DaceConfig, DaCeOrchestration
from pace.stencils.testing.grid import Grid
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
from pace.util.null_comm import NullComm
import gt4py.config
import os


def fill_corners_x(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
    with computation(PARALLEL), interval(...):
        utmp = corners.fill_corners_3cells_mult_x(
            utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        ua = corners.fill_corners_2cells_mult_x(
            ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )


def setup(dycore_config, comm):
    partitioner = util.CubedSpherePartitioner(
        util.TilePartitioner(dycore_config.layout)
    )
    communicator = util.CubedSphereCommunicator(comm, partitioner)
    grid = Grid.from_namelist(dycore_config, global_rank, backend)
    dace_config = DaceConfig(communicator, backend, DaCeOrchestration.Python)
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend, rebuild=False, validate_args=False, dace_config=dace_config,
    )
    stencil_factory = StencilFactory(
        config=stencil_config, grid_indexing=grid.grid_indexing,
    )
    metric_terms = MetricTerms.from_tile_sizing(
        npx=dycore_config.npx,
        npy=dycore_config.npy,
        npz=dycore_config.npz,
        communicator=communicator,
        backend=backend,
    )
    origin_edges = grid.grid_indexing.origin_compute(add=(-3, -3, 0))
    domain_edges = grid.grid_indexing.domain_compute(add=(6, 6, 0))
    ax_offsets_edges = grid.grid_indexing.axis_offsets(origin_edges, domain_edges)
    _fill_corners_x = stencil_factory.from_origin_domain(
        func=fill_corners_x,
        externals=ax_offsets_edges,
        origin=origin_edges,
        domain=domain_edges,
    )
    return _fill_corners_x


if __name__ == "__main__":
    if MPI is not None:
        comm = MPI.COMM_WORLD
        global_rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        global_rank = 0
        size = 1

    backend = "gt:cpu_ifirst"
    namelist = f90nml.read("/home/tobiasw/Desktop/sandbox/54_data/input.nml")
    dycore_config = DynamicalCoreConfig.from_f90nml(namelist)

    sub_tiles = dycore_config.layout[0] * dycore_config.layout[1]
    iterations = math.ceil(sub_tiles / size)
    # gt4py.config.cache_settings["root_path"] = os.environ.get("GT_CACHE_DIR_NAME", ".")
    gt4py.config.cache_settings["root_path"] = "."
    for iteration in range(iterations):
        top_tile_rank = global_rank + size * iteration
        mpi_comm = NullComm(
            rank=top_tile_rank, total_ranks=6 * sub_tiles, fill_value=0.0,
        )
        if top_tile_rank < sub_tiles:
            gt4py.config.cache_settings[
                "dir_name"
            ] = f".gt_cache_{mpi_comm.Get_rank():06}"
            obj = setup(dycore_config=dycore_config, comm=mpi_comm)
            print(f"rank {global_rank} compile for target rank {top_tile_rank}")
    if comm is not None:
        comm.Barrier()
    if global_rank == 0:
        print("compilation done")

    # if global_rank == 0:
    #     print("running 1")
    # _utmp = utils.make_storage_from_shape(
    #     grid.grid_indexing.max_shape, grid.grid_indexing.origin_full(), backend=backend,
    # )
    # _vtmp = utils.make_storage_from_shape(
    #     grid.grid_indexing.max_shape, grid.grid_indexing.origin_full(), backend=backend,
    # )

    # input_data = np.zeros(shape=(size, size, 79))

    # for i in range(int(size / 3), 2 * int(size / 3)):
    #     for j in range(int(size / 3), 2 * int(size / 3)):
    #         input_data[i, j, 0] = 4
    # va = gt_storage.from_array(
    #     input_data,
    #     backend=backend,
    #     default_origin=grid.grid_indexing.origin_full(),
    #     shape=grid.grid_indexing.max_shape,
    #     mask=None,
    #     dtype=np.float64,
    # )
    # ua = gt_storage.zeros(
    #     backend,
    #     default_origin=grid.grid_indexing.origin_full(),
    #     shape=grid.grid_indexing.max_shape,
    #     mask=None,
    #     dtype=np.float64,
    # )

    # _fill_corners_x(_utmp, _vtmp, ua, va)

    if comm is not None:
        comm.Barrier()
    print(f"Rank {global_rank} done.")
    if global_rank == 0:
        print("SUCCESS")
