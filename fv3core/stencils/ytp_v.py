from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils import yppm
from fv3core.utils.typing import FloatField


@gtscript.function
def _get_flux(
    v: FloatField, courant: FloatField, rdy: FloatField, bl: FloatField, br: FloatField
):
    """
    Compute the y-dir flux of kinetic energy(?).

    Inputs:
        v: y-dir wind
        courant: Courant number in flux form
        rdy: 1.0 / dy
        bl: ???
        br: ???

    Returns:
        Kinetic energy flux
    """
    from __externals__ import jord

    b0 = bl + br
    cfl = courant * rdy[0, -1, 0] if courant > 0 else courant * rdy
    fx0 = yppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(jord < 8):
        tmp = yppm.get_tmp(bl, b0, br)
    else:
        tmp = 1.0

    return yppm.final_flux(courant, v, fx0, tmp)


def _compute_stencil(
    courant: FloatField,
    v: FloatField,
    flux: FloatField,
    dy: FloatField,
    dya: FloatField,
    rdy: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start, jord, namelist

    with computation(PARALLEL), interval(...):

        if __INLINED(jord < 8):
            al = yppm.compute_al(v, dy)

            bl = al[0, 0, 0] - v[0, 0, 0]
            br = al[0, 1, 0] - v[0, 0, 0]

            # Zero corners
            with horizontal(
                region[i_start, j_start - 1 : j_start + 1],
                region[i_start, j_end : j_end + 2],
                region[i_end + 1, j_start - 1 : j_start + 1],
                region[i_end + 1, j_end : j_end + 2],
            ):
                bl = 0.0
                br = 0.0

        else:
            dm = yppm.dm_jord8plus(v)
            al = yppm.al_jord8plus(v, dm)

            assert __INLINED(jord == 8)
            # {
            bl, br = yppm.blbr_jord8(v, al, dm)
            # }

            assert __INLINED(namelist.grid_type < 3)
            # {
            with horizontal(region[:, j_start - 1]):
                bl, br = yppm.south_edge_jord8plus_0(v, dy, dm)

            with horizontal(region[:, j_start]):
                bl, br = yppm.south_edge_jord8plus_1(v, dy, dm)

            with horizontal(region[:, j_start + 1]):
                bl, br = yppm.south_edge_jord8plus_2(v, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(v, bl, br)

            with horizontal(region[:, j_end - 1]):
                bl, br = yppm.north_edge_jord8plus_0(v, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(v, bl, br)

            with horizontal(region[:, j_end]):
                bl, br = yppm.north_edge_jord8plus_1(v, dy, dm)

            with horizontal(region[:, j_end + 1]):
                bl, br = yppm.north_edge_jord8plus_2(v, dy, dm)

            # Zero corners
            with horizontal(
                region[i_start, j_start - 1 : j_start + 1],
                region[i_start, j_end : j_end + 2],
                region[i_end + 1, j_start - 1 : j_start + 1],
                region[i_end + 1, j_end : j_end + 2],
            ):
                bl = 0.0
                br = 0.0
            # }

        flux = _get_flux(v, courant, rdy, bl, br)


def compute(c: FloatField, v: FloatField, flux: FloatField):
    """
    Compute flux of kinetic energy in y-dir.

    Args:
        c (in): Courant number in flux form
        v (in): y-dir wind on Arakawa D-grid
        flux (out): Flux of kinetic energy
    """
    grid = spec.grid
    jord = spec.namelist.hord_mt
    if jord not in (5, 6, 7, 8):
        raise NotImplementedError(
            "Currently ytp_v is only supported for hord_mt == 5,6,7,8"
        )

    stencil = gtstencil(
        definition=_compute_stencil,
        externals={
            "jord": jord,
            "mord": jord,
            "xt_minmax": False,
        },
    )
    stencil(
        c,
        v,
        flux,
        grid.dy,
        grid.dya,
        grid.rdy,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(1, 1, 0)),
    )
