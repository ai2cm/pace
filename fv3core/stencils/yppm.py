#!/usr/bin/env python3
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import floor_cap, sign


input_vars = ["q", "c"]
inputs_params = ["jord", "ifirst", "ilast"]
output_vars = ["flux"]
# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

s11 = 11.0 / 14.0
s14 = 4.0 / 7.0
s15 = 3.0 / 14.0

sd = utils.sd
origin = (0, 2, 0)


def grid():
    return spec.grid


@gtstencil(externals={"p1": p1, "p2": p2})
def main_al_ord_under8(q: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_0(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_1(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = 0.5 * (
            (
                (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
                - dya[0, -1, 0] * q[0, -2, 0]
            )
            / (dya[0, -2, 0] + dya[0, -1, 0])
            + (
                (2.0 * dya[0, 0, 0] + dya[0, 1, 0]) * q[0, 0, 0]
                - dya[0, 0, 0] * q[0, 1, 0]
            )
            / (dya[0, 0, 0] + dya[0, 1, 0])
        )


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_2(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]


@gtstencil()
def floor_cap(var: sd, floor_value: float):
    with computation(PARALLEL), interval(0, None):
        var[0, 0, 0] = var if var > floor_value else floor_value


@gtscript.function
def get_bl(al, q):
    bl = al - q
    return bl


@gtscript.function
def get_br(al, q):
    br = al[0, 1, 0] - q
    return br


@gtscript.function
def get_b0(bl, br):
    b0 = bl + br
    return b0


@gtscript.function
def is_smt5_mord5(bl, br):
    return bl * br < 0


@gtscript.function
def is_smt5_most_mords(bl, br, b0):
    return (3.0 * abs(b0)) < abs(bl - br)


@gtscript.function
def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[0, -1, 0] - c * b0[0, -1, 0])


@gtscript.function
def fx1_c_negative(c, bl, b0):
    return (1.0 + c) * (bl + c * b0)


@gtscript.function
def final_flux(c, q, fx1, tmp):
    return q[0, -1, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)


@gtscript.function
def flux_intermediates(q, al, mord):
    bl = get_bl(al=al, q=q)
    br = get_br(al=al, q=q)
    b0 = get_b0(bl=bl, br=br)
    smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
    tmp = smt5[0, -1, 0] + smt5 * (smt5[0, -1, 0] == 0)
    return bl, br, b0, tmp


@gtscript.function
def get_flux(q, c, al, mord):
    # this does not work  -- too many layers of gtscript.function fails
    # bl, br, b0, tmp = flux_intermediates(q, al, mord)
    bl = get_bl(al=al, q=q)
    br = get_br(al=al, q=q)
    b0 = get_b0(bl=bl, br=br)
    smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
    tmp = smt5[0, -1, 0] + smt5 * (smt5[0, -1, 0] == 0)
    fx1 = fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)
    flux = (
        q[0, -1, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp
    )  # final_flux(c, q, fx1, tmp)
    return flux


@gtstencil()
def get_flux_stencil_ord6(q: sd, c: sd, al: sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(0, None):
        bl, br, b0, tmp = flux_intermediate_ord6(q, al, mord)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = final_flux(c, q, fx1, tmp)  # noqa


# TODO: remove when validated
@gtstencil()
def get_flux_stencil(q: sd, c: sd, al: sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(0, None):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        fx1 = fx1_fn(c, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx1, tmp)  # noqa
        # flux = get_flux(q, c, al, mord)
        # ...Ended up adding 1 to y direction of  storage shape with the
        # addition of  fx1[0,0,0] * tmp[0,0,0], e.g. the smt5 conditional.
        # otherwise the flux[:,je+1,:] ended up being 0.0.
        # This might need to be revisited
        # if (c > 0.0):
        #    fx1 = (1.0 - c) * (br[0, -1, 0] - c * b0[0, -1, 0])
        #    tmp = q[0, 2, 0]
        # else:
        #    fx1 = (1.0 + c) * (bl[0, 1, 0] + c * b0[0, 1, 0])
        #    tmp = q
        # if (smt5 or smt5[0, -1, 0]):
        #    flux = tmp + fx1
        # else:
        #    flux = tmp


@gtstencil()
def finalflux_ord8plus(q: sd, c: sd, bl: sd, br: sd, flux: sd):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = q[0, -1, 0] + fx1 if c > 0.0 else q + fx1


@gtstencil()
def dm_jord8plus(q: sd, al: sd, dm: sd):
    with computation(PARALLEL), interval(...):
        xt = 0.25 * (q[0, 1, 0] - q[0, -1, 0])
        dqr = max(max(q, q[0, -1, 0]), q[0, 1, 0]) - q
        dql = q - min(min(q, q[0, -1, 0]), q[0, 1, 0])
        dm = sign(min(min(abs(xt), dqr), dql), xt)


@gtstencil()
def al_jord8plus(q: sd, al: sd, dm: sd, r3: float):
    with computation(PARALLEL), interval(...):
        al = 0.5 * (q[0, -1, 0] + q) + r3 * (dm[0, -1, 0] - dm)


@gtstencil()
def blbr_jord8(q: sd, al: sd, bl: sd, br: sd, dm: sd):
    with computation(PARALLEL), interval(...):
        xt = 2.0 * dm
        aldiff = al - q
        aldiffj = al[0, 1, 0] - q
        bl = -1.0 * sign(min(abs(xt), abs(aldiff)), xt)
        br = sign(min(abs(xt), abs(aldiffj)), xt)


@gtscript.function
def xt_dya_edge_0_base(q, dya):
    return 0.5 * (
        ((2.0 * dya + dya[0, -1, 0]) * q - dya * q[0, -1, 0]) / (dya[0, -1, 0] + dya)
        + ((2.0 * dya[0, 1, 0] + dya[0, 2, 0]) * q[0, 1, 0] - dya[0, 1, 0] * q[0, 2, 0])
        / (dya[0, 1, 0] + dya[0, 2, 0])
    )


@gtscript.function
def xt_dya_edge_1_base(q, dya):
    return 0.5 * (
        (
            (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
            - dya[0, -1, 0] * q[0, -2, 0]
        )
        / (dya[0, -2, 0] + dya[0, -1, 0])
        + ((2.0 * dya + dya[0, 1, 0]) * q - dya * q[0, 1, 0]) / (dya + dya[0, 1, 0])
    )


@gtscript.function
def xt_dya_edge_0(q, dya, xt_minmax):
    xt = xt_dya_edge_0_base(q, dya)
    if xt_minmax:
        minq = min(min(min(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        maxq = max(max(max(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dya_edge_1(q, dya, xt_minmax):
    xt = xt_dya_edge_1_base(q, dya)
    if xt_minmax:
        minq = min(min(min(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        maxq = max(max(max(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtstencil()
def south_edge_jord8plus_0(q: sd, dya: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        bl = s14 * dm[0, -1, 0] + s11 * (q[0, -1, 0] - q)
        xt = xt_dya_edge_0(q, dya, xt_minmax)
        br = xt - q


@gtstencil()
def south_edge_jord8plus_1(q: sd, dya: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dya_edge_1(q, dya, xt_minmax)
        bl = xt - q
        xt = s15 * q + s11 * q[0, 1, 0] - s14 * dm[0, 1, 0]
        br = xt - q


@gtstencil()
def south_edge_jord8plus_2(q: sd, dya: sd, dm: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        xt = s15 * q[0, -1, 0] + s11 * q - s14 * dm
        bl = xt - q
        br = al[0, 1, 0] - q


@gtstencil()
def north_edge_jord8plus_0(q: sd, dya: sd, dm: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        bl = al - q
        xt = s15 * q[0, 1, 0] + s11 * q + s14 * dm
        br = xt - q


@gtstencil()
def north_edge_jord8plus_1(q: sd, dya: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = s15 * q + s11 * q[0, -1, 0] + s14 * dm[0, -1, 0]
        bl = xt - q
        xt = xt_dya_edge_0(q, dya, xt_minmax)
        br = xt - q


@gtstencil()
def north_edge_jord8plus_2(q: sd, dya: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dya_edge_1(q, dya, xt_minmax)
        bl = xt - q
        br = s11 * (q[0, 1, 0] - q) - s14 * dm[0, 1, 0]


@gtstencil()
def pert_ppm_positive_definite_constraint(a0: sd, al: sd, ar: sd, r12: float):
    with computation(PARALLEL), interval(...):
        da1 = 0.0
        a4 = 0.0
        fmin = 0.0
        if a0 <= 0.0:
            al = 0.0
            ar = 0.0
        else:
            a4 = -3.0 * (ar + al)
            da1 = ar - al
            if abs(da1) < -a4:
                fmin = a0 + 0.25 / a4 * da1 ** 2 + a4 * r12
                if fmin < 0.0:
                    if ar > 0.0 and al > 0.0:
                        ar = 0.0
                        al = 0.0
                    elif da1 > 0.0:
                        ar = -2.0 * al
                else:
                    al = -2.0 * ar


@gtstencil()
def pert_ppm_standard_constraint(a0: sd, al: sd, ar: sd):
    with computation(PARALLEL), interval(...):
        da1 = 0.0
        da2 = 0.0
        a6da = 0.0
        if al * ar < 0.0:
            da1 = al - ar
            da2 = da1 ** 2
            a6da = 3.0 * (al + ar) * da1
            if a6da < -da2:
                ar = -2.0 * al
            elif a6da > da2:
                al = -2.0 * ar
        else:
            # effect of dm=0 included here
            al = 0.0
            ar = 0.0


def compute_al(q, dyvar, jord, ifirst, ilast, js1, je3, kstart=0, nk=None):
    if nk is None:
        nk = grid().npz - kstart
    dimensions = q.shape
    local_origin = (origin[0], origin[1], kstart)
    al = utils.make_storage_from_shape(dimensions, local_origin)
    if jord < 8:
        main_al_ord_under8(
            q,
            al,
            origin=(ifirst, js1, kstart),
            domain=(ilast - ifirst + 1, je3 - js1 + 1, nk),
        )
        x_edge_domain = (dimensions[0], 1, nk)
        if not grid().nested and spec.namelist.grid_type < 3:
            # South Edge
            if grid().south_edge:
                al_x_under8_edge_0(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().js - 1, kstart),
                    domain=x_edge_domain,
                )
                al_x_under8_edge_1(
                    q, dyvar, al, origin=(0, grid().js, kstart), domain=x_edge_domain
                )
                al_x_under8_edge_2(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().js + 1, kstart),
                    domain=x_edge_domain,
                )
            # North Edge
            if grid().north_edge:
                al_x_under8_edge_0(
                    q, dyvar, al, origin=(0, grid().je, kstart), domain=x_edge_domain
                )
                al_x_under8_edge_1(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().je + 1, kstart),
                    domain=x_edge_domain,
                )
                al_x_under8_edge_2(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().je + 2, kstart),
                    domain=x_edge_domain,
                )
        if jord < 0:
            floor_cap(
                al,
                0.0,
                origin=(ifirst, grid().js - 1, kstart),
                domain=(ilast - ifirst + 1, grid().njc + 3, nk),
            )

    return al


def compute_blbr_ord8plus(q, jord, dya, ifirst, ilast, js1, je1, kstart, nk):
    r3 = 1.0 / 3.0
    grid = spec.grid
    local_origin = (origin[0], origin[1], kstart)
    bl = utils.make_storage_from_shape(q.shape, local_origin)
    br = utils.make_storage_from_shape(q.shape, local_origin)
    dm = utils.make_storage_from_shape(q.shape, local_origin)
    al = utils.make_storage_from_shape(q.shape, local_origin)
    di = ilast - ifirst + 1
    dm_jord8plus(
        q, al, dm, origin=(ifirst, grid.js - 2, kstart), domain=(di, grid.njc + 4, nk)
    )
    al_jord8plus(
        q, al, dm, r3, origin=(ifirst, js1, kstart), domain=(di, je1 - js1 + 2, nk)
    )
    if jord == 8:
        blbr_jord8(
            q,
            al,
            bl,
            br,
            dm,
            origin=(ifirst, js1, kstart),
            domain=(di, je1 - js1 + 1, nk),
        )
    else:
        raise Exception("Unimplemented jord=" + str(jord))

    if spec.namelist.grid_type < 3 and not (grid.nested or spec.namelist.regional):
        x_edge_domain = (di, 1, nk)
        do_xt_minmax = True
        if grid.south_edge:
            south_edge_jord8plus_0(
                q,
                dya,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(ifirst, grid.js - 1, kstart),
                domain=x_edge_domain,
            )
            south_edge_jord8plus_1(
                q,
                dya,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(ifirst, grid.js, kstart),
                domain=x_edge_domain,
            )
            south_edge_jord8plus_2(
                q,
                dya,
                dm,
                al,
                bl,
                br,
                origin=(ifirst, grid.js + 1, kstart),
                domain=x_edge_domain,
            )
            pert_ppm(q, bl, br, 1, ifirst, grid.js - 1, kstart, di, 3, nk)
        if grid.north_edge:
            north_edge_jord8plus_0(
                q,
                dya,
                dm,
                al,
                bl,
                br,
                origin=(ifirst, grid.je - 1, kstart),
                domain=x_edge_domain,
            )
            north_edge_jord8plus_1(
                q,
                dya,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(ifirst, grid.je, kstart),
                domain=x_edge_domain,
            )
            north_edge_jord8plus_2(
                q,
                dya,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(ifirst, grid.je + 1, kstart),
                domain=x_edge_domain,
            )
            pert_ppm(q, bl, br, 1, ifirst, grid.je - 1, kstart, di, 3, nk)
        return bl, br


def compute_flux(q, c, flux, jord, ifirst, ilast, kstart=0, nk=None):
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    js1 = grid.js + 2 if grid.south_edge else grid.js - 1
    je3 = grid.je - 1 if grid.north_edge else grid.je + 2
    je1 = grid.je - 2 if grid.north_edge else grid.je + 1
    mord = abs(jord)
    if mord not in [5, 6, 7, 8]:
        raise Exception(
            "We have only implemented yppm for hord=5, 6, 7, and 8, not " + str(jord)
        )
    flux_origin = (ifirst, grid.js, kstart)
    flux_domain = (ilast - ifirst + 1, grid.njc + 1, nk)
    if mord < 8:
        al = compute_al(q, grid.dya, jord, ifirst, ilast, js1, je3, kstart, nk)
        get_flux_stencil(
            q, c, al, flux, mord=mord, origin=flux_origin, domain=flux_domain
        )
    else:
        bl, br = compute_blbr_ord8plus(
            q, jord, grid.dya, ifirst, ilast, js1, je1, kstart, nk
        )
        finalflux_ord8plus(q, c, bl, br, flux, origin=flux_origin, domain=flux_domain)


# Optimized PPM in perturbation form:
def pert_ppm(a0, al, ar, iv, istart, jstart, kstart, ni, nj, nk):
    r12 = 1.0 / 12.0
    if iv == 0:
        pert_ppm_positive_definite_constraint(
            a0, al, ar, r12, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    else:
        pert_ppm_standard_constraint(
            a0, al, ar, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    return al, ar
