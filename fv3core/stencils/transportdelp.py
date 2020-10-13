import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.corners import fill_4corners


sd = utils.sd


@gtstencil()
def hydro_x_fluxes(delp: sd, pt: sd, utc: sd, fx: sd, fx1: sd):
    with computation(PARALLEL), interval(...):
        fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
        fx = pt[-1, 0, 0] if utc > 0.0 else pt
        fx1 = utc * fx1
        fx = fx1 * fx


@gtstencil()
def hydro_y_fluxes(delp: sd, pt: sd, vtc: sd, fy: sd, fy1: sd):
    with computation(PARALLEL), interval(...):
        fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
        fy = pt[0, -1, 0] if vtc > 0.0 else pt
        fy1 = vtc * fy1
        fy = fy1 * fy


@gtstencil()
def nonhydro_x_fluxes(delp: sd, pt: sd, w: sd, utc: sd, fx: sd, fx1: sd, fx2: sd):
    with computation(PARALLEL), interval(...):
        fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
        fx = pt[-1, 0, 0] if utc > 0.0 else pt
        fx2 = w[-1, 0, 0] if utc > 0.0 else w
        fx1 = utc * fx1
        fx = fx1 * fx
        fx2 = fx1 * fx2


@gtstencil()
def nonhydro_y_fluxes(delp: sd, pt: sd, w: sd, vtc: sd, fy: sd, fy1: sd, fy2: sd):
    with computation(PARALLEL), interval(...):
        fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
        fy = pt[0, -1, 0] if vtc > 0.0 else pt
        fy2 = w[0, -1, 0] if vtc > 0.0 else w
        fy1 = vtc * fy1
        fy = fy1 * fy
        fy2 = fy1 * fy2


@gtstencil()
def transportdelp_hydrostatic(
    delp: sd, pt: sd, fx: sd, fx1: sd, fy: sd, fy1: sd, rarea: sd, delpc: sd, ptc: sd
):
    with computation(PARALLEL), interval(...):
        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc


@gtstencil()
def transportdelp_nonhydrostatic(
    delp: sd,
    pt: sd,
    w: sd,
    fx: sd,
    fx1: sd,
    fx2: sd,
    fy: sd,
    fy1: sd,
    fy2: sd,
    rarea: sd,
    delpc: sd,
    ptc: sd,
    wc: sd,
):
    with computation(PARALLEL), interval(...):
        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc


def compute(delp, pt, w, utc, vtc, wc):
    grid = spec.grid
    orig = (grid.is_ - 1, grid.js - 1, 0)
    hydrostatic = int(spec.namelist.hydrostatic)

    fx = utils.make_storage_from_shape(delp.shape, origin=orig)
    fx1 = utils.make_storage_from_shape(pt.shape, origin=orig)
    fy = utils.make_storage_from_shape(delp.shape, origin=orig)
    fy1 = utils.make_storage_from_shape(w.shape, origin=orig)
    delpc = utils.make_storage_from_shape(delp.shape, origin=orig)
    ptc = utils.make_storage_from_shape(pt.shape, origin=orig)

    # TODO: untested currently, don't have serialized data
    if hydrostatic:
        if spec.namelist.grid_type < 3 and not grid.nested:
            fill_4corners(delp, "x", grid)
            fill_4corners(pt, "x", grid)
        hydro_x_fluxes(
            delp,
            pt,
            utc,
            fx,
            fx1,
            origin=orig,
            domain=(grid.nic + 3, grid.njc + 2, grid.npz),
        )
        if spec.namelist.grid_type < 3 and not grid.nested:
            fill_4corners(delp, "y", grid)
            fill_4corners(pt, "y", grid)
        hydro_y_fluxes(
            delp,
            pt,
            utc,
            fy,
            fy1,
            origin=orig,
            domain=(grid.nic + 2, grid.njc + 3, grid.npz),
        )

        transportdelp_hydrostatic(
            delp,
            pt,
            fx,
            fx1,
            fy,
            fy1,
            spec.grid.rarea,
            delpc,
            ptc,
            origin=orig,
            domain=(grid.nic + 2, grid.njc + 2, grid.npz),
        )

    else:
        if spec.namelist.grid_type < 3 and not grid.nested:
            fill_4corners(delp, "x", grid)
            fill_4corners(pt, "x", grid)
            fill_4corners(w, "x", grid)
        fx2 = utils.make_storage_from_shape(w.shape, origin=orig)
        nonhydro_x_fluxes(
            delp,
            pt,
            w,
            utc,
            fx,
            fx1,
            fx2,
            origin=orig,
            domain=(grid.nic + 3, grid.njc + 2, grid.npz),
        )
        if spec.namelist.grid_type < 3 and not grid.nested:
            fill_4corners(delp, "y", grid)
            fill_4corners(pt, "y", grid)
            fill_4corners(w, "y", grid)
        fy2 = utils.make_storage_from_shape(w.shape, origin=orig)
        nonhydro_y_fluxes(
            delp,
            pt,
            w,
            vtc,
            fy,
            fy1,
            fy2,
            origin=orig,
            domain=(grid.nic + 2, grid.njc + 3, grid.npz),
        )

        transportdelp_nonhydrostatic(
            delp,
            pt,
            w,
            fx,
            fx1,
            fx2,
            fy,
            fy1,
            fy2,
            spec.grid.rarea,
            delpc,
            ptc,
            wc,
            origin=orig,
            domain=(grid.nic + 2, grid.njc + 2, grid.npz),
        )

    return delpc, ptc
