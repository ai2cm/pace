import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


def grid():
    return spec.grid


sd = utils.sd
stencil_corner = True


@gtstencil()
def main_ut(uc: sd, vc: sd, cosa_u: sd, rsin_u: sd, ut: sd):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            uc - 0.25 * cosa_u * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
        ) * rsin_u


@gtstencil()
def ut_y_edge(uc: sd, sin_sg1: sd, sin_sg3: sd, ut: sd, *, dt: float):
    with computation(PARALLEL), interval(0, -1):
        ut[0, 0, 0] = (uc / sin_sg3[-1, 0, 0]) if (uc * dt > 0) else (uc / sin_sg1)


@gtstencil()
def ut_x_edge(uc: sd, cosa_u: sd, vt: sd, ut: sd):
    with computation(PARALLEL), interval(0, -1):
        ut[0, 0, 0] = uc - 0.25 * cosa_u * (
            vt[-1, 0, 0] + vt[0, 0, 0] + vt[-1, 1, 0] + vt[0, 1, 0]
        )


@gtstencil()
def main_vt(uc: sd, vc: sd, cosa_v: sd, rsin_v: sd, vt: sd):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            vc - 0.25 * cosa_v * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
        ) * rsin_v


@gtstencil()
def vt_y_edge(vc: sd, cosa_v: sd, ut: sd, vt: sd):
    with computation(PARALLEL), interval(0, -1):
        vt[0, 0, 0] = vc - 0.25 * cosa_v * (
            ut[0, -1, 0] + ut[1, -1, 0] + ut[0, 0, 0] + ut[1, 0, 0]
        )


@gtstencil()
def vt_x_edge(vc: sd, sin_sg2: sd, sin_sg4: sd, vt: sd, *, dt: float):
    with computation(PARALLEL), interval(0, -1):
        vt[0, 0, 0] = (vc / sin_sg4[0, -1, 0]) if (vc * dt > 0) else (vc / sin_sg2)


@gtscript.function
def ra_x_func(area, xfx_adv):
    return area + xfx_adv - xfx_adv[1, 0, 0]


@gtstencil()
def xfx_adv_stencil(
    ut: sd,
    rdxa: sd,
    area: sd,
    dy: sd,
    sin_sg1: sd,
    sin_sg3: sd,
    crx_adv: sd,
    xfx_adv: sd,
    ra_x: sd,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        xfx_adv[0, 0, 0] = dt * ut
        crx_adv[0, 0, 0] = xfx_adv * rdxa[-1, 0, 0] if xfx_adv > 0 else xfx_adv * rdxa
        xfx_adv[0, 0, 0] = (
            dy * xfx_adv * sin_sg3[-1, 0, 0] if xfx_adv > 0 else dy * xfx_adv * sin_sg1
        )
        ra_x = ra_x_func(area, xfx_adv)


@gtscript.function
def ra_y_func(area, yfx_adv):
    return area + yfx_adv - yfx_adv[0, 1, 0]


@gtstencil()
def yfx_adv_stencil(
    vt: sd,
    rdya: sd,
    area: sd,
    dx: sd,
    sin_sg2: sd,
    sin_sg4: sd,
    cry_adv: sd,
    yfx_adv: sd,
    ra_y: sd,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        yfx_adv[0, 0, 0] = dt * vt
        cry_adv[0, 0, 0] = yfx_adv * rdya[0, -1, 0] if yfx_adv > 0 else yfx_adv * rdya
        yfx_adv[0, 0, 0] = (
            dx * yfx_adv * sin_sg4[0, -1, 0] if yfx_adv > 0 else dx * yfx_adv * sin_sg2
        )
        ra_y = ra_y_func(area, yfx_adv)


def compute(uc_in, vc_in, ut_in, vt_in, xfx_adv, yfx_adv, crx_adv, cry_adv, dt):
    ut = compute_ut(uc_in, vc_in, grid().cosa_u, grid().rsin_u, ut_in)
    vt = compute_vt(
        uc_in,
        vc_in,
        grid().cosa_v,
        grid().rsin_v,
        grid().sin_sg2,
        grid().sin_sg4,
        vt_in,
    )
    update_ut_y_edge(uc_in, grid().sin_sg1, grid().sin_sg3, ut, dt)
    update_vt_y_edge(vc_in, grid().cosa_v, ut, vt)
    update_vt_x_edge(vc_in, grid().sin_sg2, grid().sin_sg4, vt, dt)
    update_ut_x_edge(uc_in, grid().cosa_u, vt, ut)
    corner_shape = (1, 1, uc_in.shape[2])
    if grid().sw_corner:
        sw_corner(uc_in, vc_in, ut, vt, grid().cosa_u, grid().cosa_v, corner_shape)
    if grid().se_corner:
        se_corner(uc_in, vc_in, ut, vt, grid().cosa_u, grid().cosa_v, corner_shape)
    if grid().ne_corner:
        ne_corner(uc_in, vc_in, ut, vt, grid().cosa_u, grid().cosa_v, corner_shape)
    if grid().nw_corner:
        nw_corner(uc_in, vc_in, ut, vt, grid().cosa_u, grid().cosa_v, corner_shape)
    ra_x = utils.make_storage_from_shape(uc_in.shape, grid().compute_x_origin())
    xfx_adv_stencil(
        ut,
        grid().rdxa,
        grid().area,
        grid().dy,
        grid().sin_sg1,
        grid().sin_sg3,
        crx_adv,
        xfx_adv,
        ra_x,
        dt,
        origin=grid().compute_x_origin(),
        domain=grid().domain_y_compute_xbuffer(),
    )
    ra_y = utils.make_storage_from_shape(vc_in.shape, grid().compute_y_origin())
    yfx_adv_stencil(
        vt,
        grid().rdya,
        grid().area,
        grid().dx,
        grid().sin_sg2,
        grid().sin_sg4,
        cry_adv,
        yfx_adv,
        ra_y,
        dt,
        origin=grid().compute_y_origin(),
        domain=grid().domain_x_compute_ybuffer(),
    )
    # TODO remove the need for a copied extra ut and vt variables, edit in place (rexolve issue with data getting zeroed out)
    ut_in[:, :, :] = ut[:, :, :]
    vt_in[:, :, :] = vt[:, :, :]
    return ra_x, ra_y


def compute_ut(uc_in, vc_in, cosa_u, rsin_u, ut_in):
    ut_origin = (grid().is_ - 1, grid().jsd, 0)
    ut = utils.make_storage_from_shape(ut_in.shape, ut_origin)
    main_ut(
        uc_in,
        vc_in,
        cosa_u,
        rsin_u,
        ut,
        origin=ut_origin,
        domain=(grid().nic + 3, grid().njd, grid().npz),
    )
    ut[: grid().is_ - 1, :, :] = ut_in[: grid().is_ - 1, :, :]
    ut[grid().ie + 3 :, :, :] = ut_in[grid().ie + 3 :, :, :]
    # fill in for j /=2 and j/=3
    if grid().south_edge:
        ut[:, grid().js - 1 : grid().js + 1, :] = ut_in[
            :, grid().js - 1 : grid().js + 1, :
        ]
    # fill in for j/=npy-1 and j /= npy
    if grid().north_edge:
        ut[:, grid().je : grid().je + 2, :] = ut_in[:, grid().je : grid().je + 2, :]
    return ut


def update_ut_y_edge(uc, sin_sg1, sin_sg3, ut, dt):
    edge_shape = (1, ut.shape[1], ut.shape[2])
    if grid().west_edge:
        ut_y_edge(
            uc,
            sin_sg1,
            sin_sg3,
            ut,
            dt=dt,
            origin=(grid().is_, 0, 0),
            domain=edge_shape,
        )
    if grid().east_edge:
        ut_y_edge(
            uc,
            sin_sg1,
            sin_sg3,
            ut,
            dt=dt,
            origin=(grid().ie + 1, 0, 0),
            domain=edge_shape,
        )


def update_ut_x_edge(uc, cosa_u, vt, ut):
    i1 = grid().is_ + 2 if grid().west_edge else grid().is_
    i2 = grid().ie - 1 if grid().east_edge else grid().ie + 1
    edge_shape = (i2 - i1 + 1, 2, ut.shape[2])
    if grid().south_edge:
        ut_x_edge(uc, cosa_u, vt, ut, origin=(i1, grid().js - 1, 0), domain=edge_shape)
    if grid().north_edge:
        ut_x_edge(uc, cosa_u, vt, ut, origin=(i1, grid().je, 0), domain=edge_shape)


def compute_vt(uc_in, vc_in, cosa_v, rsin_v, sin_sg2, sin_sg4, vt_in):
    vt_origin = (grid().isd, grid().js - 1, 0)
    vt = utils.make_storage_from_shape(vt_in.shape, vt_origin)
    main_vt(
        uc_in,
        vc_in,
        cosa_v,
        rsin_v,
        vt,
        origin=vt_origin,
        domain=(grid().nid, grid().njc + 3, grid().npz),
    )  # , origin=(0, 2, 0), domain=(vt.shape[0]-1, main_j_size, vt.shape[2]))
    # cannot pass vt_in array to stencil without it zeroing out data outside specified domain
    # So... for now copying in so the 'undefined' answers match
    vt[:, : grid().js - 1, :] = vt_in[:, : grid().js - 1, :]
    vt[:, grid().je + 3, :] = vt_in[:, grid().je + 3, :]
    if grid().south_edge:
        vt[:, grid().js, :] = vt_in[:, grid().js, :]
    if grid().north_edge:
        vt[:, grid().je + 1, :] = vt_in[:, grid().je + 1, :]
    return vt


def update_vt_y_edge(vc, cosa_v, ut, vt):
    if grid().west_edge or grid().east_edge:
        j1 = grid().js + 2 if grid().south_edge else grid().js
        j2 = grid().je if grid().north_edge else grid().je + 2
        edge_shape = (2, j2 - j1, ut.shape[2])
        if grid().west_edge:
            vt_y_edge(
                vc, cosa_v, ut, vt, origin=(grid().is_ - 1, j1, 0), domain=edge_shape
            )
        if grid().east_edge:
            vt_y_edge(vc, cosa_v, ut, vt, origin=(grid().ie, j1, 0), domain=edge_shape)


def update_vt_x_edge(vc, sin_sg2, sin_sg4, vt, dt):
    if grid().south_edge or grid().north_edge:
        edge_shape = (vt.shape[0], 1, vt.shape[2])
        if grid().south_edge:
            vt_x_edge(
                vc,
                sin_sg2,
                sin_sg4,
                vt,
                dt=dt,
                origin=(0, grid().js, 0),
                domain=edge_shape,
            )
        if grid().north_edge:
            vt_x_edge(
                vc,
                sin_sg2,
                sin_sg4,
                vt,
                dt=dt,
                origin=(0, grid().je + 1, 0),
                domain=edge_shape,
            )


# -------------------- CORNERS-----------------


def corner_ut_stencil(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    from __externals__ import ux, uy, vi, vj, vx, vy

    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            (
                uc[0, 0, 0]
                - 0.25
                * cosa_u[0, 0, 0]
                * (
                    vt[vi, vy, 0]
                    + vt[vx, vy, 0]
                    + vt[vx, vj, 0]
                    + vc[vi, vj, 0]
                    - 0.25
                    * cosa_v[vi, vj, 0]
                    * (ut[ux, 0, 0] + ut[ux, uy, 0] + ut[0, uy, 0])
                )
            )
            * 1.0
            / (1.0 - 0.0625 * cosa_u[0, 0, 0] * cosa_v[vi, vj, 0])
        )


# for the non-stencil version of filling corners
def get_damp(cosa_u, cosa_v, ui, uj, vi, vj):
    return 1.0 / (1.0 - 0.0625 * cosa_u[ui, uj, :] * cosa_v[vi, vj, :])


def index_offset(lower, u, south=True):
    if lower == u:
        offset = 1
    else:
        offset = -1
    if south:
        offset *= -1
    return offset


def corner_ut(
    uc,
    vc,
    ut,
    vt,
    cosa_u,
    cosa_v,
    ui,
    uj,
    vi,
    vj,
    west,
    lower,
    south=True,
    vswitch=False,
):
    if vswitch:
        lowerfactor = 1 if lower else -1
    else:
        lowerfactor = 1
    vx = vi + index_offset(west, False, south) * lowerfactor
    ux = ui + index_offset(west, True, south) * lowerfactor
    vy = vj + index_offset(lower, False, south) * lowerfactor
    uy = uj + index_offset(lower, True, south) * lowerfactor
    if stencil_corner:
        decorator = gtscript.stencil(
            backend=utils.backend,
            externals={
                "vi": vi - ui,
                "vj": vj - uj,
                "ux": ux - ui,
                "uy": uy - uj,
                "vx": vx - ui,
                "vy": vy - uj,
            },
            rebuild=utils.rebuild,
        )
        corner_stencil = decorator(corner_ut_stencil)
        corner_stencil(
            uc,
            vc,
            ut,
            vt,
            cosa_u,
            cosa_v,
            origin=(ui, uj, 0),
            domain=(1, 1, grid().npz),
        )
    else:
        damp = get_damp(cosa_u, cosa_v, ui, uj, vi, vj)
        ut[ui, uj, :] = (
            uc[ui, uj, :]
            - 0.25
            * cosa_u[ui, uj, :]
            * (
                vt[vi, vy, :]
                + vt[vx, vy, :]
                + vt[vx, vj, :]
                + vc[vi, vj, :]
                - 0.25
                * cosa_v[vi, vj, :]
                * (ut[ux, uj, :] + ut[ux, uy, :] + ut[ui, uy, :])
            )
        ) * damp


def sw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    t = grid().is_ + 1
    n = grid().is_
    z = grid().is_ - 1
    corner_ut(uc, vc, ut, vt, cosa_u, cosa_v, t, z, n, z, west=True, lower=True)
    corner_ut(
        vc, uc, vt, ut, cosa_v, cosa_u, z, t, z, n, west=True, lower=True, vswitch=True
    )
    corner_ut(uc, vc, ut, vt, cosa_u, cosa_v, t, n, n, t, west=True, lower=False)
    corner_ut(
        vc, uc, vt, ut, cosa_v, cosa_u, n, t, t, n, west=True, lower=False, vswitch=True
    )


def se_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    t = grid().js + 1
    n = grid().js
    z = grid().js - 1
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        grid().ie,
        z,
        grid().ie,
        z,
        west=False,
        lower=True,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        grid().ie + 1,
        t,
        grid().ie + 2,
        n,
        west=False,
        lower=True,
        vswitch=True,
    )
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        grid().ie,
        n,
        grid().ie,
        t,
        west=False,
        lower=False,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        grid().ie,
        t,
        grid().ie,
        n,
        west=False,
        lower=False,
        vswitch=True,
    )


def ne_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        grid().ie,
        grid().je + 1,
        grid().ie,
        grid().je + 2,
        west=False,
        lower=False,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        grid().ie + 1,
        grid().je,
        grid().ie + 2,
        grid().je,
        west=False,
        lower=False,
        south=False,
        vswitch=True,
    )
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        grid().ie,
        grid().je,
        grid().ie,
        grid().je,
        west=False,
        lower=True,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        grid().ie,
        grid().je,
        grid().ie,
        grid().je,
        west=False,
        lower=True,
        south=False,
        vswitch=True,
    )


def nw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    t = grid().js + 1
    n = grid().js
    z = grid().js - 1
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        t,
        grid().je + 1,
        n,
        grid().je + 2,
        west=True,
        lower=False,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        z,
        grid().je,
        z,
        grid().je,
        west=True,
        lower=False,
        south=False,
        vswitch=True,
    )
    corner_ut(
        uc,
        vc,
        ut,
        vt,
        cosa_u,
        cosa_v,
        t,
        grid().je,
        n,
        grid().je,
        west=True,
        lower=True,
    )
    corner_ut(
        vc,
        uc,
        vt,
        ut,
        cosa_v,
        cosa_u,
        n,
        grid().je,
        t,
        grid().je,
        west=True,
        lower=True,
        south=False,
        vswitch=True,
    )


# TODO Probably can delete -- but in case we want to do analysis to show it doesn't matter at all
"""
def sw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    west_corner_ut_lowest(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ + 1, grid().js - 1, 0), domain=corner_shape)
    west_corner_ut_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ + 1, grid().js, 0), domain=corner_shape)
    south_corner_vt_left(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ - 1, grid().js + 1, 0), domain=corner_shape)
    south_corner_vt_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_, grid().js + 1, 0), domain=corner_shape)


def se_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    east_corner_ut_lowest(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().js - 1, 0), domain=corner_shape)
    east_corner_ut_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().js, 0), domain=corner_shape)
    south_corner_vt_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie + 1, grid().js + 1, 0), domain=corner_shape)
    south_corner_vt_left(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().js + 1, 0), domain=corner_shape)

def ne_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    east_corner_ut_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().je + 1, 0), domain=corner_shape)
    east_corner_ut_lowest(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().je, 0), domain=corner_shape)
    north_corner_vt_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie + 1, grid().je, 0), domain=corner_shape)
    north_corner_vt_left(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().ie, grid().je, 0), domain=corner_shape)

def nw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
    west_corner_ut_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ + 1, grid().je+1, 0), domain=corner_shape)
    west_corner_ut_lowest(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ + 1, grid().je, 0), domain=corner_shape)
    north_corner_vt_left(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_ - 1, grid().je, 0), domain=corner_shape)
    north_corner_vt_adjacent(uc, vc, ut, vt, cosa_u, cosa_v, origin=(grid().is_, grid().je, 0), domain=corner_shape)


@gtstencil()
def west_corner_ut_lowest(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_u = 1. / (1.0 - 0.0625 * cosa_u[0, 0, 0] * cosa_v[-1, 0, 0])
        ut[0, 0, 0] = (uc[0, 0, 0]-0.25 * cosa_u[0, 0, 0] * (vt[-1, 1, 0] + vt[0, 1, 0] + vt[0, 0, 0] + vc[-1, 0, 0] -
                                                             0.25 * cosa_v[-1, 0, 0] * (ut[-1, 0, 0] + ut[-1, -1, 0] +
                                                                                        ut[0, -1, 0]))) * damp_u


@gtstencil()
def west_corner_ut_adjacent(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp = 1. / (1. - 0.0625 * cosa_u[0, 0, 0] * cosa_v[-1, 1, 0])
        ut[0, 0, 0] = (uc[0, 0, 0] - 0.25 * cosa_u[0, 0, 0] * (vt[-1, 0, 0] + vt[0, 0, 0] + vt[0, 1, 0] + vc[-1, 1, 0] -
                                                               0.25 * cosa_v[-1, 1, 0] * (ut[-1, 0, 0] + ut[-1, 1, 0] +
                                                                                          ut[0, 1, 0]))) * damp


@gtstencil()
def south_corner_vt_left(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_v = 1. / (1.0 - 0.0625 * cosa_u[0, -1, 0] * cosa_v[0, 0, 0])
        vt[0, 0, 0] = (vc[0, 0, 0] - 0.25 * cosa_v[0, 0, 0] * (ut[1, -1, 0] + ut[1, 0, 0] + ut[0, 0, 0] + uc[0, -1, 0] -
                                                               0.25 * cosa_u[0, -1, 0] *
                                                               (vt[0, -1, 0] + vt[-1, -1, 0] + vt[-1, 0, 0]))) * damp_v


@gtstencil()
def south_corner_vt_adjacent(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_v = 1. / (1.0 - 0.0625 * cosa_u[1, -1, 0] * cosa_v[0, 0, 0])
        vt[0, 0, 0] = (vc[0, 0, 0] - 0.25 * cosa_v[0, 0, 0] * (ut[0, -1, 0] + ut[0, 0, 0] + ut[1, 0, 0] + uc[1, -1, 0] -
                                                               0.25 * cosa_u[1, -1, 0] *
                                                               (vt[0, -1, 0] + vt[1, -1, 0] + vt[1, 0, 0]))) * damp_v


@gtstencil()
def east_corner_ut_lowest(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_u = 1. / (1.0 - 0.0625 * cosa_u[0, 0, 0] * cosa_v[0, 0, 0])
        ut[0, 0, 0] = (uc[0, 0, 0]-0.25 * cosa_u[0, 0, 0] * (vt[0, 1, 0] + vt[-1, 1, 0] + vt[-1, 0, 0] + vc[0, 0, 0] -
                                                             0.25 * cosa_v[0, 0, 0] * (ut[1, 0, 0] + ut[1, -1, 0] +
                                                                                       ut[0, -1, 0]))) * damp_u


@gtstencil()
def east_corner_ut_adjacent(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp = 1. / (1. - 0.0625 * cosa_u[0, 0, 0] * cosa_v[0, 1, 0])
        ut[0, 0, 0] = (uc[0, 0, 0] - 0.25 * cosa_u[0, 0, 0] * (vt[0, 0, 0] + vt[-1, 0, 0] + vt[-1, 1, 0] + vc[0, 1, 0] -
                                                               0.25 * cosa_v[0, 1, 0] * (ut[1, 0, 0] + ut[1, 1, 0] +
                                                                                         ut[0, 1, 0]))) * damp


@gtstencil()
def north_corner_vt_left(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_v = 1. / (1.0 - 0.0625 * cosa_u[0, 0, 0] * cosa_v[0, 0, 0])
        vt[0, 0, 0] = (vc[0, 0, 0] - 0.25 * cosa_v[0, 0, 0] * (ut[1, 0, 0] + ut[1, -1, 0] + ut[0, -1, 0] + uc[0, 0, 0] -
                                                               0.25 * cosa_u[0, 0, 0] *
                                                               (vt[0, 1, 0] + vt[-1, 1, 0] + vt[-1, 0, 0]))) * damp_v

@gtstencil()
def north_corner_vt_adjacent(uc: sd, vc: sd, ut: sd, vt: sd, cosa_u: sd, cosa_v: sd):
    with computation(PARALLEL), interval(...):
        damp_v = 1. / (1.0 - 0.0625 * cosa_u[1, 0, 0] * cosa_v[0, 0, 0])
        vt[0, 0, 0] = (vc[0, 0, 0] - 0.25 * cosa_v[0, 0, 0] * (ut[0, 0, 0] + ut[0, -1, 0] + ut[1, -1, 0] + uc[1, 0, 0] -
                                                               0.25 * cosa_u[1, 0, 0] *
                                                               (vt[0, 1, 0] + vt[1, 1, 0] + vt[1, 0, 0]))) * damp_v
"""
