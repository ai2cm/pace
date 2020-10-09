from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.xppm as xppm
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil

from .xppm import (
    compute_al,
    final_flux,
    flux_intermediates,
    fx1_fn,
    get_b0,
    get_bl,
    get_br,
    is_smt5_mord5,
    is_smt5_most_mords,
    s11,
    s14,
    s15,
    xt_dxa_edge_0_base,
    xt_dxa_edge_1_base,
)


sd = utils.sd


@gtstencil()
def get_flux_u_stencil_old(q: sd, c: sd, al: sd, rdx: sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(...):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
        fx0 = fx1_fn(cfl, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx0, tmp)  # noqa


@gtstencil()
def get_flux_u_stencil(
    q: sd, c: sd, al: sd, rdx: sd, bl: sd, br: sd, flux: sd, mord: int
):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl=bl, br=br)
        smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
        cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
        fx0 = fx1_fn(cfl, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx0, tmp)  # noqa


@gtstencil()
def get_flux_u_ord8plus(q: sd, c: sd, rdx: sd, bl: sd, br: sd, flux: sd):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
        fx1 = fx1_fn(cfl, br, b0, bl)
        flux = q[-1, 0, 0] + fx1 if c > 0.0 else q + fx1


@gtstencil()
def br_bl_main(q: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        bl = get_bl(al=al, q=q)  # noqa
        br = get_br(al=al, q=q)  # noqa


@gtstencil()
def br_bl_corner(br: sd, bl: sd):
    with computation(PARALLEL), interval(...):
        bl = 0
        br = 0


def zero_br_bl_corners_west(br, bl):
    grid = spec.grid
    corner_domain = (2, 1, grid.npz)
    if grid.sw_corner:
        br_bl_corner(br, bl, origin=(grid.is_ - 1, grid.js, 0), domain=corner_domain)
    if grid.nw_corner:
        br_bl_corner(
            br, bl, origin=(grid.is_ - 1, grid.je + 1, 0), domain=corner_domain
        )


def zero_br_bl_corners_east(br, bl):
    grid = spec.grid
    corner_domain = (2, 1, grid.npz)
    if grid.se_corner:
        br_bl_corner(br, bl, origin=(grid.ie, grid.js, 0), domain=corner_domain)
    if grid.ne_corner:
        br_bl_corner(br, bl, origin=(grid.ie, grid.je + 1, 0), domain=corner_domain)


def compute(c, u, v, flux):
    # This is an input argument in the Fortran code, but is never called with anything but this namelist option
    grid = spec.grid
    iord = spec.namelist.hord_mt
    if iord not in [5, 6, 7, 8]:
        raise Exception("Currently ytp_v is only supported for hord_mt == 5, 6, 7, 8")
    is3 = grid.is_ - 1  # max(5, grid.is_ - 1)
    ie3 = grid.ie + 1  # min(grid.npx - 1, grid.ie+1)
    tmp_origin = (is3, grid.js, 0)
    bl = utils.make_storage_from_shape(v.shape, tmp_origin)
    br = utils.make_storage_from_shape(v.shape, tmp_origin)
    if iord < 8:
        al = compute_al(u, grid.dx, iord, is3, ie3 + 1, grid.js, grid.je + 1)
        # get_flux_u_stencil_old(u, c, al, grid.rdx, flux, iord, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc + 1, grid.npz))
        br_bl_main(
            u,
            al,
            bl,
            br,
            origin=(is3, grid.js, 0),
            domain=(ie3 - is3 + 1, grid.njc + 1, grid.npz),
        )
        zero_br_bl_corners_west(br, bl)
        zero_br_bl_corners_east(br, bl)

        get_flux_u_stencil(
            u,
            c,
            al,
            grid.rdx,
            bl,
            br,
            flux,
            iord,
            origin=grid.compute_origin(),
            domain=(grid.nic + 1, grid.njc + 1, grid.npz),
        )
    else:
        is1 = grid.is_ + 2 if grid.west_edge else grid.is_ - 1
        ie1 = grid.ie - 2 if grid.east_edge else grid.ie + 1
        dm = utils.make_storage_from_shape(u.shape, grid.compute_origin())
        al = utils.make_storage_from_shape(u.shape, grid.compute_origin())
        dj = grid.njc + 1
        jfirst = grid.js
        kstart = 0
        nk = grid.npz
        r3 = 1.0 / 3.0
        xppm.dm_iord8plus(
            u,
            al,
            dm,
            origin=(grid.is_ - 2, jfirst, kstart),
            domain=(grid.nic + 4, dj, nk),
        )
        xppm.al_iord8plus(
            u, al, dm, r3, origin=(is1, jfirst, kstart), domain=(ie1 - is1 + 2, dj, nk)
        )
        if iord == 8:
            xppm.blbr_iord8(
                u,
                al,
                bl,
                br,
                dm,
                origin=(is1, jfirst, kstart),
                domain=(ie1 - is1 + 2, dj, nk),
            )
        else:
            raise Exception("Unimplemented iord=" + str(iord))

        if spec.namelist.grid_type < 3 and not (grid.nested or spec.namelist.regional):
            y_edge_domain = (1, dj, nk)
            do_xt_minmax = False
            if grid.west_edge:
                xppm.west_edge_iord8plus_0(
                    u,
                    grid.dx,
                    dm,
                    bl,
                    br,
                    do_xt_minmax,
                    origin=(grid.is_ - 1, jfirst, kstart),
                    domain=y_edge_domain,
                )
                xppm.west_edge_iord8plus_1(
                    u,
                    grid.dx,
                    dm,
                    bl,
                    br,
                    do_xt_minmax,
                    origin=(grid.is_, jfirst, kstart),
                    domain=y_edge_domain,
                )
                xppm.west_edge_iord8plus_2(
                    u,
                    grid.dx,
                    dm,
                    al,
                    bl,
                    br,
                    origin=(grid.is_ + 1, jfirst, kstart),
                    domain=y_edge_domain,
                )
                zero_br_bl_corners_west(br, bl)
                xppm.pert_ppm(u, bl, br, -1, grid.is_ + 1, jfirst, kstart, 1, dj, nk)

            if grid.east_edge:
                xppm.east_edge_iord8plus_0(
                    u,
                    grid.dx,
                    dm,
                    al,
                    bl,
                    br,
                    origin=(grid.ie - 1, jfirst, kstart),
                    domain=y_edge_domain,
                )
                xppm.east_edge_iord8plus_1(
                    u,
                    grid.dx,
                    dm,
                    bl,
                    br,
                    do_xt_minmax,
                    origin=(grid.ie, jfirst, kstart),
                    domain=y_edge_domain,
                )
                xppm.east_edge_iord8plus_2(
                    u,
                    grid.dx,
                    dm,
                    bl,
                    br,
                    do_xt_minmax,
                    origin=(grid.ie + 1, jfirst, kstart),
                    domain=y_edge_domain,
                )
                zero_br_bl_corners_east(br, bl)
                xppm.pert_ppm(u, bl, br, -1, grid.ie - 1, jfirst, kstart, 1, dj, nk)
        get_flux_u_ord8plus(
            u,
            c,
            grid.rdx,
            bl,
            br,
            flux,
            origin=(grid.is_, grid.js, kstart),
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )


# TODO merge better with equivalent xppm functions, the main difference is there is no minmax on xt here


@gtstencil()
def west_edge_iord8plus_0(q: sd, dxa: sd, dm: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        bl = s14 * dm[-1, 0, 0] + s11 * (q[-1, 0, 0] - q)
        xt = xt_dxa_edge_0_base(q, dxa)
        br = xt - q


@gtstencil()
def west_edge_iord8plus_1(q: sd, dxa: sd, dm: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1_base(q, dxa)
        bl = xt - q
        xt = s15 * q + s11 * q[1, 0, 0] - s14 * dm[1, 0, 0]
        br = xt - q


@gtstencil()
def east_edge_iord8plus_1(q: sd, dxa: sd, dm: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        xt = s15 * q + s11 * q[-1, 0, 0] + s14 * dm[-1, 0, 0]
        bl = xt - q
        xt = xt_dxa_edge_0_base(q, dxa)
        br = xt - q


@gtstencil()
def east_edge_iord8plus_2(q: sd, dxa: sd, dm: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1_base(q, dxa)
        bl = xt - q
        br = s11 * (q[1, 0, 0] - q) - s14 * dm[1, 0, 0]
