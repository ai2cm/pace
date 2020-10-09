import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.yppm as yppm
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil

from .yppm import (
    compute_al,
    final_flux,
    fx1_fn,
    get_b0,
    get_bl,
    get_br,
    is_smt5_mord5,
    is_smt5_most_mords,
    s11,
    s14,
    s15,
    xt_dya_edge_0_base,
    xt_dya_edge_1_base,
)


sd = utils.sd


@gtstencil()
def get_flux_v_stencil(
    q: sd, c: sd, al: sd, rdy: sd, bl: sd, br: sd, flux: sd, mord: int
):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl=bl, br=br)
        smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        tmp = smt5[0, -1, 0] + smt5 * (smt5[0, -1, 0] == 0)
        cfl = c * rdy[0, -1, 0] if c > 0 else c * rdy
        fx0 = fx1_fn(cfl, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx0, tmp)  # noqa


@gtstencil()
def get_flux_v_ord8plus(q: sd, c: sd, rdy: sd, bl: sd, br: sd, flux: sd):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        cfl = c * rdy[0, -1, 0] if c > 0 else c * rdy
        fx1 = fx1_fn(cfl, br, b0, bl)
        flux = q[0, -1, 0] + fx1 if c > 0.0 else q + fx1


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


def zero_br_bl_corners_south(br, bl):
    grid = spec.grid
    corner_domain = (1, 2, grid.npz)
    if grid.sw_corner:
        br_bl_corner(br, bl, origin=(grid.is_, grid.js - 1, 0), domain=corner_domain)
    if grid.se_corner:
        br_bl_corner(br, bl, origin=(grid.ie + 1, grid.js - 1, 0), domain=corner_domain)


def zero_br_bl_corners_north(br, bl):
    grid = spec.grid
    corner_domain = (1, 2, grid.npz)
    if grid.nw_corner:
        br_bl_corner(br, bl, origin=(grid.is_, grid.je, 0), domain=corner_domain)
    if grid.ne_corner:
        br_bl_corner(br, bl, origin=(grid.ie + 1, grid.je, 0), domain=corner_domain)


def compute(c, u, v, flux):
    grid = spec.grid
    # This is an input argument in the Fortran code, but is never called with anything but this namelist option
    jord = spec.namelist.hord_mt
    if jord not in [5, 6, 7, 8]:
        raise Exception("Currently ytp_v is only supported for hord_mt == 5,6,7,8")
    js3 = grid.js - 1
    je3 = grid.je + 1

    tmp_origin = (grid.is_, grid.js - 1, 0)
    bl = utils.make_storage_from_shape(u.shape, tmp_origin)
    br = utils.make_storage_from_shape(u.shape, tmp_origin)

    if jord < 8:
        # this not get the exact right edges
        al = compute_al(v, grid.dy, jord, grid.is_, grid.ie + 1, js3, je3 + 1)
        br_bl_main(
            v,
            al,
            bl,
            br,
            origin=(grid.is_, grid.js - 1, 0),
            domain=(grid.nic + 1, grid.njc + 2, grid.npz),
        )
        zero_br_bl_corners_south(br, bl)
        zero_br_bl_corners_north(br, bl)
        get_flux_v_stencil(
            v,
            c,
            al,
            grid.rdy,
            bl,
            br,
            flux,
            jord,
            origin=(grid.is_, grid.js, 0),
            domain=(grid.nic + 1, grid.njc + 1, grid.npz),
        )
    else:
        js1 = grid.js + 2 if grid.south_edge else grid.js - 1
        je1 = grid.je - 2 if grid.north_edge else grid.je + 1
        dm = utils.make_storage_from_shape(v.shape, grid.compute_origin())
        al = utils.make_storage_from_shape(v.shape, grid.compute_origin())
        di = grid.nic + 1
        ifirst = grid.is_
        kstart = 0
        nk = grid.npz
        r3 = 1.0 / 3.0
        yppm.dm_jord8plus(
            v,
            al,
            dm,
            origin=(ifirst, grid.js - 2, kstart),
            domain=(di, grid.njc + 4, nk),
        )
        yppm.al_jord8plus(
            v, al, dm, r3, origin=(ifirst, js1, kstart), domain=(di, je1 - js1 + 2, nk)
        )
        if jord == 8:
            yppm.blbr_jord8(
                v,
                al,
                bl,
                br,
                dm,
                origin=(ifirst, js1, kstart),
                domain=(di, je1 - js1 + 2, nk),
            )
        else:
            raise Exception("Unimplemented jord=" + str(jord))

        if spec.namelist.grid_type < 3 and not (grid.nested or spec.namelist.regional):
            x_edge_domain = (di, 1, nk)
            do_xt_minmax = False
            if grid.south_edge:
                yppm.south_edge_jord8plus_0(
                    v,
                    grid.dy,
                    dm,
                    bl,
                    br,
                    False,
                    origin=(ifirst, grid.js - 1, kstart),
                    domain=x_edge_domain,
                )
                yppm.south_edge_jord8plus_1(
                    v,
                    grid.dy,
                    dm,
                    bl,
                    br,
                    False,
                    origin=(ifirst, grid.js, kstart),
                    domain=x_edge_domain,
                )
                yppm.south_edge_jord8plus_2(
                    v,
                    grid.dy,
                    dm,
                    al,
                    bl,
                    br,
                    origin=(ifirst, grid.js + 1, kstart),
                    domain=x_edge_domain,
                )
                zero_br_bl_corners_south(br, bl)
                yppm.pert_ppm(v, bl, br, -1, ifirst, grid.js + 1, kstart, di, 1, nk)

            if grid.north_edge:
                yppm.north_edge_jord8plus_0(
                    v,
                    grid.dy,
                    dm,
                    al,
                    bl,
                    br,
                    origin=(ifirst, grid.je - 1, kstart),
                    domain=x_edge_domain,
                )
                yppm.north_edge_jord8plus_1(
                    v,
                    grid.dy,
                    dm,
                    bl,
                    br,
                    False,
                    origin=(ifirst, grid.je, kstart),
                    domain=x_edge_domain,
                )
                yppm.north_edge_jord8plus_2(
                    v,
                    grid.dy,
                    dm,
                    bl,
                    br,
                    False,
                    origin=(ifirst, grid.je + 1, kstart),
                    domain=x_edge_domain,
                )
                zero_br_bl_corners_north(br, bl)
                yppm.pert_ppm(v, bl, br, -1, ifirst, grid.je - 1, kstart, di, 1, nk)
        get_flux_v_ord8plus(
            v,
            c,
            grid.rdy,
            bl,
            br,
            flux,
            origin=(grid.is_, grid.js, kstart),
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )
