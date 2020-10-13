#!/usr/bin/env python3
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.d2a2c_vect as d2a2c
import fv3core.stencils.divergence_corner as divergence_corner
import fv3core.stencils.ke_c_sw as ke_c_sw
import fv3core.stencils.transportdelp as transportdelp
import fv3core.stencils.vorticitytransport_cgrid as vorticity_transport
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def geoadjust_ut(ut: sd, dy: sd, sin_sg3: sd, sin_sg1: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


@gtstencil()
def geoadjust_vt(vt: sd, dx: sd, sin_sg4: sd, sin_sg2: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1, 0] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


@gtstencil()
def absolute_vorticity(vort: sd, fC: sd, rarea_c: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtstencil()
def circulation_cgrid(uc: sd, vc: sd, dxc: sd, dyc: sd, vort_c: sd):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)
        vort_c: C-grid vorticity (output)
    """
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        fx = dxc * uc
        fy = dyc * vc

        vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy

        with parallel(region[i_start, j_start], region[i_start, j_end + 1]):
            vort_c += fy[-1, 0, 0]

        with parallel(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
            vort_c -= fy[0, 0, 0]


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    fx = utils.make_storage_from_shape(delp.shape, origin_halo1)
    fy = utils.make_storage_from_shape(delp.shape, origin_halo1)
    d2a2c.compute(dord4, uc, vc, u, v, ua, va, ut, vt)
    if spec.namelist.nord > 0:
        divergence_corner.compute(u, v, ua, va, divgd)
    geo_origin = (grid.is_ - 1, grid.js - 1, 0)
    geoadjust_ut(
        ut,
        grid.dy,
        grid.sin_sg3,
        grid.sin_sg1,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 3, grid.njc + 2, grid.npz),
    )
    geoadjust_vt(
        vt,
        grid.dx,
        grid.sin_sg4,
        grid.sin_sg2,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 2, grid.njc + 3, grid.npz),
    )
    delpc, ptc = transportdelp.compute(delp, pt, w, ut, vt, omga)
    ke, vort = ke_c_sw.compute(uc, vc, u, v, ua, va, dt2)
    circulation_cgrid(
        uc,
        vc,
        grid.dxc,
        grid.dyc,
        vort,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
    )
    absolute_vorticity(
        vort,
        grid.fC,
        grid.rarea_c,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    vorticity_transport.compute(uc, vc, vort, ke, v, u, fx, fy, dt2)
    return delpc, ptc
