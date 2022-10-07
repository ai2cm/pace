from typing import Dict

import numpy as np

import pace.fv3core
import pace.util
from pace.fv3core.initialization import init_baroclinic_state
from pace.util.grid import MetricTerms
from pace.util.mpi import MPIComm
from pace.util.quantity import Quantity
from util.pace.util.grid.helper import GridData


def get_cube_comm(layout, comm: MPIComm):
    return pace.util.CubedSphereCommunicator(
        comm=comm,
        partitioner=pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(layout=layout)
        ),
    )


def get_quantity_factory(layout, nx_tile, ny_tile, nz):
    nx = nx_tile // layout[0]
    ny = ny_tile // layout[1]
    return pace.util.QuantityFactory(
        sizer=pace.util.SubtileGridSizer(
            nx=nx, ny=ny, nz=nz, n_halo=3, extra_dim_lengths={}
        ),
        numpy=np,
    )


def metric_terms_to_quantity_dict(metric_terms: MetricTerms) -> Dict[str, Quantity]:
    return {
        "grid": metric_terms.grid,
        "agrid": metric_terms.agrid,
        "area": metric_terms.area,
        "area_c": metric_terms.area_c,
        "dx": metric_terms.dx,
        "dy": metric_terms.dy,
        "dxa": metric_terms.dxa,
        "dya": metric_terms.dya,
        "dxc": metric_terms.dxc,
        "dyc": metric_terms.dyc,
        "ec1": metric_terms.ec1,
        "ec2": metric_terms.ec2,
        "ew1": metric_terms.ew1,
        "ew2": metric_terms.ew2,
        "cos_sg1": metric_terms.cos_sg1,
        "cos_sg2": metric_terms.cos_sg2,
        "cos_sg3": metric_terms.cos_sg3,
        "cos_sg4": metric_terms.cos_sg4,
        "cos_sg5": metric_terms.cos_sg5,
        "cos_sg6": metric_terms.cos_sg6,
        "cos_sg7": metric_terms.cos_sg7,
        "cos_sg8": metric_terms.cos_sg8,
        "cos_sg9": metric_terms.cos_sg9,
        "sin_sg1": metric_terms.sin_sg1,
        "sin_sg2": metric_terms.sin_sg2,
        "sin_sg3": metric_terms.sin_sg3,
        "sin_sg4": metric_terms.sin_sg4,
        "sin_sg5": metric_terms.sin_sg5,
        "sin_sg6": metric_terms.sin_sg6,
        "sin_sg7": metric_terms.sin_sg7,
        "sin_sg8": metric_terms.sin_sg8,
        "sin_sg9": metric_terms.sin_sg9,
        "rarea_c": metric_terms.rarea_c,
        "rarea": metric_terms.rarea,
        "rdx": metric_terms.rdx,
        "rdy": metric_terms.rdy,
        "rdxa": metric_terms.rdxa,
        "rdya": metric_terms.rdya,
        "rdxc": metric_terms.rdxc,
        "rdyc": metric_terms.rdyc,
        "cosa": metric_terms.cosa,
        "sina": metric_terms.sina,
        "rsina": metric_terms.rsina,
        "rsin2": metric_terms.rsin2,
        "l2c_v": metric_terms.l2c_v,
        "l2c_u": metric_terms.l2c_u,
        "es1": metric_terms.es1,
        "es2": metric_terms.es2,
        "ee1": metric_terms.ee1,
        "ee2": metric_terms.ee2,
        "rsin_u": metric_terms.rsin_u,
        "rsin_v": metric_terms.rsin_v,
        "cosa_u": metric_terms.cosa_u,
        "cosa_v": metric_terms.cosa_v,
        "cosa_s": metric_terms.cosa_s,
        "sina_u": metric_terms.sina_u,
        "sina_v": metric_terms.sina_v,
        "divg_u": metric_terms.divg_u,
        "divg_v": metric_terms.divg_v,
        "del6_u": metric_terms.del6_u,
        "del6_v": metric_terms.del6_v,
        "vlon": metric_terms.vlon,
        "vlat": metric_terms.vlat,
        "z11": metric_terms.z11,
        "z12": metric_terms.z12,
        "z21": metric_terms.z21,
        "z22": metric_terms.z22,
        "a11": metric_terms.a11,
        "a12": metric_terms.a12,
        "a21": metric_terms.a21,
        "a22": metric_terms.a22,
        # Can't test these because they are only computed on edge ranks,
        # but get broadcast through the entire compute domain, so they are
        # technically grid-dependent. They can't be computed on all ranks
        # because they need the lat/lon data of the tile edge.
        # "edge_w": metric_terms.edge_w,
        # "edge_e": metric_terms.edge_e,
        # "edge_s": metric_terms.edge_s,
        # "edge_n": metric_terms.edge_n,
        # "edge_vect_w": metric_terms.edge_vect_w,
        # "edge_vect_w_2d": metric_terms.edge_vect_w_2d,
        # "edge_vect_e": metric_terms.edge_vect_e,
        # "edge_vect_e_2d": metric_terms.edge_vect_e_2d,
        # "edge_vect_s": metric_terms.edge_vect_s,
        # "edge_vect_n": metric_terms.edge_vect_n,
    }


def dycore_state_to_quantity_dict(
    dycore_state: pace.fv3core.DycoreState,
) -> Dict[str, Quantity]:
    return {
        "u": dycore_state.u,
        "v": dycore_state.u,
        "delp": dycore_state.delp,
        "delz": dycore_state.delz,
        "pt": dycore_state.pt,
        "ps": dycore_state.ps,
        "peln": dycore_state.peln,
        "pk": dycore_state.pk,
        "pkz": dycore_state.pkz,
        "pe": dycore_state.pe,
        "phis": dycore_state.phis,
        "w": dycore_state.w,
        "qvapor": dycore_state.qvapor,
    }


def gather_all(
    quantity_dict: Dict[str, Quantity], tile_comm: pace.util.TileCommunicator
) -> Dict[str, Quantity]:
    gathered = {}
    for name, quantity in quantity_dict.items():
        gathered[name] = tile_comm.gather(quantity)
    return gathered


def test_grid_init_not_decomposition_dependent():
    nx_tile, ny_tile, nz = 48, 48, 5
    # use all ranks for 3x3 decomposition of single tile
    # we can use TileCommunicator for halo updates
    comm_3by3 = MPIComm()
    global_rank = comm_3by3.Get_rank()
    cube_comm = get_cube_comm(layout=(3, 3), comm=comm_3by3)
    metric_terms_3by3 = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(3, 3), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )
    computed_3by3 = metric_terms_to_quantity_dict(metric_terms_3by3)
    gathered_3by3 = gather_all(computed_3by3, cube_comm.tile)
    # only need 6 ranks for 1x1 decomposition, should be root rank of each tile
    # so we can easily gather on each tile
    compute_1by1 = cube_comm.tile.rank == 0
    comm_1by1 = comm_3by3.Split(color=int(compute_1by1), key=global_rank)
    if compute_1by1:
        metric_terms_1by1 = MetricTerms(
            quantity_factory=get_quantity_factory(
                layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
            ),
            communicator=get_cube_comm(layout=(1, 1), comm=comm_1by1),
        )
        computed_1by1 = metric_terms_to_quantity_dict(metric_terms_1by1)
        for name in computed_1by1:
            assert allclose(computed_1by1[name], gathered_3by3[name], name, global_rank)


def test_baroclinic_init_not_decomposition_dependent():
    nx_tile, ny_tile, nz = 24, 24, 79
    # use all ranks for 3x3 decomposition of single tile
    # we can use TileCommunicator for halo updates
    comm_3by3 = MPIComm()
    global_rank = comm_3by3.Get_rank()
    cube_comm_3by3 = get_cube_comm(layout=(3, 3), comm=comm_3by3)
    quantity_factory_3by3 = get_quantity_factory(
        layout=(3, 3), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
    )
    metric_terms_3by3 = MetricTerms(
        quantity_factory=quantity_factory_3by3,
        communicator=cube_comm_3by3,
    )
    grid_data_3by3 = GridData.new_from_metric_terms(metric_terms_3by3)
    state_3by3 = init_baroclinic_state(
        grid_data=grid_data_3by3,
        quantity_factory=quantity_factory_3by3,
        adiabatic=False,
        hydrostatic=False,
        moist_phys=True,
        comm=cube_comm_3by3,
    )
    computed_3by3 = dycore_state_to_quantity_dict(state_3by3)
    gathered_3by3 = gather_all(computed_3by3, cube_comm_3by3.tile)
    # only need 6 ranks for 1x1 decomposition, should be root rank of each tile
    # so we can easily gather on each tile
    compute_1by1 = cube_comm_3by3.tile.rank == 0
    comm_1by1 = comm_3by3.Split(color=int(compute_1by1), key=global_rank)
    if compute_1by1:
        cube_comm_1by1 = get_cube_comm(layout=(1, 1), comm=comm_1by1)
        quantity_factory_1by1 = get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        )
        metric_terms_1by1 = MetricTerms(
            quantity_factory=quantity_factory_1by1,
            communicator=cube_comm_1by1,
        )
        grid_data_1by1 = GridData.new_from_metric_terms(metric_terms_1by1)
        state_1by1 = init_baroclinic_state(
            grid_data=grid_data_1by1,
            quantity_factory=quantity_factory_1by1,
            adiabatic=False,
            hydrostatic=False,
            moist_phys=True,
            comm=cube_comm_1by1,
        )
        computed_1by1 = dycore_state_to_quantity_dict(state_1by1)
        for name in computed_1by1:
            assert allclose(computed_1by1[name], gathered_3by3[name], name, global_rank)


def allclose(q_1by1: pace.util.Quantity, q_3by3: pace.util.Quantity, name: str, rank):
    print("1by1", q_1by1.metadata, "3by3", q_3by3.metadata)
    assert q_1by1.view[:].shape == q_3by3.view[:].shape, name
    same = (q_1by1.view[:] == q_3by3.view[:]) | np.isnan(q_1by1.view[:])
    all_same = np.all(same)
    if not all_same:
        print(np.sum(~same), np.where(~same))
        import xarray as xr

        ds = xr.Dataset(
            data_vars={
                f"{name}_1by1": (q_1by1.metadata.dims, q_1by1.view[:]),
                f"{name}_3by3": (q_3by3.metadata.dims, q_3by3.view[:]),
            }
        )
        ds.to_netcdf(f"failure_{rank}.nc")
    return all_same
