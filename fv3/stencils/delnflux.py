#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3.utils.corners as corners
import gt4py.gtscript as gtscript
import fv3.stencils.copy_stencil as cp
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd


@utils.stencil()
def fx2_order(q: sd, del6_v: sd, fx2: sd, order: int):
    with computation(PARALLEL), interval(...):
        fx2[0, 0, 0] = del6_v * (q[-1, 0, 0] - q)
        fx2[0, 0, 0] = -1.0 * fx2 if order > 1 else fx2


@utils.stencil()
def fy2_order(q: sd, del6_u: sd, fy2: sd, order: int):
    with computation(PARALLEL), interval(...):
        fy2[0, 0, 0] = del6_u * (q[0, -1, 0] - q)
        fy2[0, 0, 0] = fy2 * -1 if order > 1 else fy2


# WARNING: untested
@utils.stencil()
def fx2_firstorder_use_sg(q: sd, sin_sg1: sd, sin_sg3: sd, dy: sd, rdxc: sd, fx2: sd):
    with computation(PARALLEL), interval(...):
        fx2[0, 0, 0] = (
            0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc
        )


# WARNING: untested
@utils.stencil()
def fy2_firstorder_use_sg(q: sd, sin_sg2: sd, sin_sg4: sd, dx: sd, rdyc: sd, fy2: sd):
    with computation(PARALLEL), interval(...):
        fy2[0, 0, 0] = (
            0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc
        )


@utils.stencil()
def d2_highorder(fx2: sd, fy2: sd, rarea: sd, d2: sd):
    with computation(PARALLEL), interval(...):
        d2[0, 0, 0] = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea


@utils.stencil()
def d2_damp(q: sd, d2: sd, damp: float):
    with computation(PARALLEL), interval(...):
        d2[0, 0, 0] = damp * q


@utils.stencil()
def add_diffusive(fx: sd, fx2: sd, fy: sd, fy2: sd):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + fx2
        fy[0, 0, 0] = fy + fy2


@utils.stencil()
def diffusive_damp(fx: sd, fx2: sd, fy: sd, fy2: sd, mass: sd, damp: float):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


@utils.stencil()
def diffusive_damp_x(fx: sd, fx2: sd, mass: sd, damp: float):
    with computation(PARALLEL), interval(...):
        fx = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2


@utils.stencil()
def diffusive_damp_y(fy: sd, fy2: sd, mass: sd, damp: float):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def compute_delnflux(data, column_info):
    if "mass" not in data:
        data["mass"] = None
    # utils.compute_column_split(
    #     compute_delnflux_no_sg, data, nord_column, "nord", ["fx", "fy"], spec.grid
    # )
    raise NotImplementedError()


def compute_del6vflux(data, nord_column):
    if "mass" not in data:
        data["mass"] = None
    utils.compute_column_split(
        compute_no_sg, data, nord_column, "nord", ["fx2", "fy2", "d2", "q"], spec.grid
    )


def compute_delnflux_no_sg(q, fx, fy, nord, damp_c, d2=None, mass=None):
    grid = spec.grid
    if d2 is None:
        d2 = utils.make_storage_from_shape(q.shape, grid.default_origin())
    if damp_c <= 1e-4:
        return fx, fy
    damp = (damp_c * grid.da_min) ** (nord + 1)
    fx2 = utils.make_storage_from_shape(q.shape, grid.default_origin())
    fy2 = utils.make_storage_from_shape(q.shape, grid.default_origin())
    fx2, fy2, d2, q = compute_no_sg(q, fx2, fy2, nord, damp, d2, mass)
    diffuse_domain = grid.domain_shape_compute_buffer_2d()
    if mass is None:
        add_diffusive(
            fx, fx2, fy, fy2, origin=(grid.is_, grid.js, 0), domain=diffuse_domain
        )
    else:
        # this won't work if we end up with different sized arrays for fx and fy, would need to have different domains for each
        diffusive_damp(
            fx,
            fx2,
            fy,
            fy2,
            mass,
            damp,
            origin=(grid.is_, grid.js, 0),
            domain=diffuse_domain,
        )
        # diffusive_damp_x(fx, fx2, mass, damp, origin=(grid.is_, grid.js, 0), domaingrid.domain_shape_compute_x())
        # diffusive_damp_y(fx, fx2,  mass, damp, origin=(grid.is_, grid.js, 0), domain=grid.domain_shape.compute_y())
    return fx, fy


def compute_no_sg(q, fx2, fy2, nord, damp_c, d2, mass=None):
    grid = spec.grid
    nord = int(nord)
    i1 = grid.is_ - 1 - nord
    i2 = grid.ie + 1 + nord
    j1 = grid.js - 1 - nord
    j2 = grid.je + 1 + nord

    origin_d2 = (i1, j1, 0)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, q.shape[2])
    if mass is None:
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        d2 = cp.copy(q, origin_d2, domain=domain_d2)
    if nord > 0:
        corners.copy_corners(d2, "x", grid)
    f1_ny = grid.je - grid.js + 1 + 2 * nord
    f1_nx = grid.ie - grid.is_ + 2 + 2 * nord
    fx_origin = (grid.is_ - nord, grid.js - nord, 0)

    fx2_order(
        d2, grid.del6_v, fx2, order=1, origin=fx_origin, domain=(f1_nx, f1_ny, grid.npz)
    )

    if nord > 0:
        corners.copy_corners(d2, "y", grid)
    fy2_order(
        d2,
        grid.del6_u,
        fy2,
        order=1,
        origin=fx_origin,
        domain=(f1_nx - 1, f1_ny + 1, grid.npz),
    )

    if nord > 0:
        for n in range(nord):
            nt = nord - 1 - n
            nt_origin = (grid.is_ - nt - 1, grid.js - nt - 1, 0)
            nt_ny = grid.je - grid.js + 3 + 2 * nt
            nt_nx = grid.ie - grid.is_ + 3 + 2 * nt
            d2_highorder(
                fx2,
                fy2,
                grid.rarea,
                d2,
                origin=nt_origin,
                domain=(nt_nx, nt_ny, grid.npz),
            )

            corners.copy_corners(d2, "x", grid)
            nt_origin = (grid.is_ - nt, grid.js - nt, 0)
            fx2_order(
                d2,
                grid.del6_v,
                fx2,
                order=2 + n,
                origin=nt_origin,
                domain=(nt_nx - 1, nt_ny - 2, grid.npz),
            )

            corners.copy_corners(d2, "y", grid)

            fy2_order(
                d2,
                grid.del6_u,
                fy2,
                order=2 + n,
                origin=nt_origin,
                domain=(nt_nx - 2, nt_ny - 1, grid.npz),
            )

    return fx2, fy2, d2, q
