#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3.utils.corners as corners
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config  as spec
import fv3.stencils.yppm as yppm
import fv3.stencils.xppm as xppm
import fv3.stencils.delnflux as delnflux

origin = (0, 0, 0)
sd = utils.sd

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def q_i_stencil(q: sd, area: sd, yfx: sd, fy2:sd, ra_y:sd, q_i:sd):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i = (q * area + fyy - fyy[0, 1, 0]) / ra_y


@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def q_j_stencil(q: sd, area: sd, xfx: sd, fx2:sd, ra_x:sd, q_j:sd):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j = (q * area + fx1 - fx1[1, 0, 0]) / ra_x

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def transport_flux(f: sd, f2: sd, mf: sd):
    with computation(PARALLEL), interval(...):
        f = 0.5 * (f + f2) * mf


def compute(data, nord_column):
    for optional_arg in ['mass', 'mfx', 'mfy']:
        if optional_arg not in data:
            data[optional_arg] = None
    utils.compute_column_split(compute_no_sg, data, nord_column, 'nord', ['q', 'fx', 'fy'], grid)



def compute_no_sg(q, crx, cry, hord, xfx, yfx, ra_x, ra_y,
                  nord=None, damp_c=None, mass=None, mfx=None, mfy=None):
    grid = spec.grid
    q_i = utils.make_storage_from_shape(q.shape, (grid.isd, grid.js, 0))
    q_j = utils.make_storage_from_shape(q.shape, (grid.is_, grid.jsd, 0))
    if hord == 10:
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord
    corners.copy_corners(q, 'y', grid)
    fy2 = yppm.compute_flux(q, cry, ord_in, grid.isd, grid.ied)
    q_i_stencil(q, grid.area, yfx, fy2, ra_y, q_i, origin=(grid.isd, grid.js, 0), domain=(grid.nid, grid.njc + 1, grid.npz))
    fx = xppm.compute_flux(q_i, crx, ord_ou, grid.js, grid.je)
    corners.copy_corners(q, 'x', grid)
    fx2 = xppm.compute_flux(q, crx, ord_in, grid.jsd, grid.jed)
    q_j_stencil(q, grid.area, xfx, fx2, ra_x, q_j, origin=(grid.is_, grid.jsd, 0), domain=(grid.nic + 1, grid.njd, grid.npz))
    fy = yppm.compute_flux(q_j, cry, ord_ou, grid.is_, grid.ie)
    if mfx is not None and mfy is not None:

        transport_flux(fx, fx2, mfx, origin=grid.compute_origin(), domain=grid.domain_shape_compute_x())
        transport_flux(fy, fy2, mfy, origin=grid.compute_origin(), domain=grid.domain_shape_compute_y())
        if (mass is not None) and (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, mass=mass)
    else:

        transport_flux(fx, fx2, xfx, origin=grid.compute_origin(), domain=grid.domain_shape_compute_x())
        transport_flux(fy, fy2, yfx, origin=grid.compute_origin(), domain=grid.domain_shape_compute_y())
        if (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c)
    return q, fx, fy
