from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def q_i_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    yfx: FloatField,
    fy2: FloatField,
    ra_y: FloatField,
    q_i: FloatField,
):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i[0, 0, 0] = (q * area + fyy - fyy[0, 1, 0]) / ra_y


@gtstencil()
def q_j_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    xfx: FloatField,
    fx2: FloatField,
    ra_x: FloatField,
    q_j: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j[0, 0, 0] = (q * area + fx1 - fx1[1, 0, 0]) / ra_x


@gtscript.function
def transport_flux(f, f2, mf):
    return 0.5 * (f + f2) * mf


@gtstencil()
def transport_flux_xy(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mfx: FloatField,
    mfy: FloatField,
):
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, :-1]):
            fx = transport_flux(fx, fx2, mfx)
        with horizontal(region[:-1, :]):
            fy = transport_flux(fy, fy2, mfy)


def compute(data, nord_column):
    for optional_arg in ["mass", "mfx", "mfy"]:
        if optional_arg not in data:
            data[optional_arg] = None
    # utils.compute_column_split(
    #     compute_no_sg, data, nord_column, "nord", ["q", "fx", "fy"], grid
    # )
    raise NotImplementedError()


def compute_no_sg(
    q: FloatField,
    crx: FloatField,
    cry: FloatField,
    hord: int,
    xfx: FloatField,
    yfx: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
    fx: FloatField,
    fy: FloatField,
    kstart: int = 0,
    nk: Optional[int] = None,
    nord: Optional[float] = None,
    damp_c: Optional[float] = None,
    mass: FloatField = None,
    mfx: FloatField = None,
    mfy: FloatField = None,
):
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    kslice = slice(kstart, kstart + nk)
    compute_origin = (grid.is_, grid.js, kstart)
    q_i = utils.make_storage_from_shape(q.shape, (grid.isd, grid.js, kstart))
    q_j = utils.make_storage_from_shape(q.shape, (grid.is_, grid.jsd, kstart))
    fy2 = utils.make_storage_from_shape(q.shape, compute_origin)
    fx2 = utils.make_storage_from_shape(q.shape, compute_origin)
    if hord == 10:
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord

    xppm_object_in = xppm.XPPM(spec.namelist, ord_in)
    yppm_object_in = yppm.YPPM(spec.namelist, ord_in)
    xppm_object_ou = xppm.XPPM(spec.namelist, ord_ou)
    yppm_object_ou = yppm.YPPM(spec.namelist, ord_ou)

    corners.copy_corners_y_stencil(
        q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )
    yppm_object_in(q, cry, fy2, grid.isd, grid.ied, kstart=kstart, nk=nk)
    q_i_stencil(
        q,
        grid.area,
        yfx,
        fy2,
        ra_y,
        q_i,
        origin=(grid.isd, grid.js, kstart),
        domain=(grid.nid, grid.njc + 1, nk),
    )

    xppm_object_ou(q_i, crx, fx, grid.js, grid.je, kstart=kstart, nk=nk)
    corners.copy_corners_x_stencil(
        q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )
    xppm_object_in(q, crx, fx2, grid.jsd, grid.jed, kstart=kstart, nk=nk)
    q_j_stencil(
        q,
        grid.area,
        xfx,
        fx2,
        ra_x,
        q_j,
        origin=(grid.is_, grid.jsd, kstart),
        domain=(grid.nic + 1, grid.njd, nk),
    )
    yppm_object_ou(q_j, cry, fy, grid.is_, grid.ie, kstart=kstart, nk=nk)

    if mfx is not None and mfy is not None:
        transport_flux_xy(
            fx,
            fx2,
            fy,
            fy2,
            mfx,
            mfy,
            origin=compute_origin,
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )

        if (mass is not None) and (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(
                q, fx, fy, nord, damp_c, kstart, nk, mass=mass
            )
    else:
        transport_flux_xy(
            fx,
            fx2,
            fy,
            fy2,
            xfx,
            yfx,
            origin=compute_origin,
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )

        if (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, kstart, nk)
