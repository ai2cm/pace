#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

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
halo = utils.halo
sd = utils.sd
origin = (0, 2, 0)


def grid():
    return spec.grid


@gtscript.stencil(backend=utils.backend, externals={"p1": p1, "p2": p2})
def main_al(q: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])


@gtscript.stencil(backend=utils.backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_edge_0(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q


@gtscript.stencil(backend=utils.backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_edge_1(q: sd, dya: sd, al: sd):
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


@gtscript.stencil(backend=utils.backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_edge_2(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]


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
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value


@gtscript.function
def is_smt5_mord5(bl, br):
    return bl * br < 0


@gtscript.function
def is_smt5_most_mords(bl, br, b0):
    return (3.0 * absolute_value(in_array=b0)) < absolute_value(in_array=(bl - br))


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


# TODO: remove when validated
@gtscript.stencil(backend=utils.backend)
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


def compute_al(q, dyvar, jord, ifirst, ilast, js1, je3):
    dimensions = q.shape
    al = utils.make_storage_from_shape(dimensions, origin)
    if jord < 8:
        main_al(
            q,
            al,
            origin=(ifirst, js1, 0),
            domain=(ilast - ifirst + 1, je3 - js1 + 1, grid().npz),
        )
        x_edge_domain = (dimensions[0], 1, dimensions[2])
        if not grid().nested and spec.namelist["grid_type"] < 3:
            # South Edge
            if grid().south_edge:
                al_x_edge_0(
                    q, dyvar, al, origin=(0, grid().js - 1, 0), domain=x_edge_domain
                )
                al_x_edge_1(
                    q, dyvar, al, origin=(0, grid().js, 0), domain=x_edge_domain
                )
                al_x_edge_2(
                    q, dyvar, al, origin=(0, grid().js + 1, 0), domain=x_edge_domain
                )
            # North Edge
            if grid().north_edge:
                al_x_edge_0(
                    q, dyvar, al, origin=(0, grid().je, 0), domain=x_edge_domain
                )
                al_x_edge_1(
                    q, dyvar, al, origin=(0, grid().je + 1, 0), domain=x_edge_domain
                )
                al_x_edge_2(
                    q, dyvar, al, origin=(0, grid().je + 2, 0), domain=x_edge_domain
                )
    return al


def compute_flux(q, c, jord, ifirst, ilast):
    js1 = max(5, grid().js - 1)
    je3 = min(grid().npy, grid().je + 2)
    al = compute_al(q, grid().dya, jord, ifirst, ilast, js1, je3)
    mord = abs(jord)
    flux = utils.make_storage_from_shape(q.shape, origin)
    flux_domain = (ilast - ifirst + 1, grid().njc + 1, grid().npz)
    get_flux_stencil(
        q, c, al, flux, mord=mord, origin=(ifirst, grid().js, 0), domain=flux_domain
    )
    return flux
