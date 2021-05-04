from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    compile_assert,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import FrozenStencil
from fv3core.stencils import yppm
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def _get_flux(
    v: FloatField,
    courant: FloatField,
    rdy: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
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
    cfl = courant * rdy[0, -1] if courant > 0 else courant * rdy[0, 0]
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
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    rdy: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, j_end, j_start, jord

    with computation(PARALLEL), interval(...):
        if __INLINED(jord < 8):
            al = yppm.compute_al(v, dy)

            bl = al[0, 0, 0] - v[0, 0, 0]
            br = al[0, 1, 0] - v[0, 0, 0]

        else:
            dm = yppm.dm_jord8plus(v)
            al = yppm.al_jord8plus(v, dm)

            compile_assert(jord == 8)

            bl, br = yppm.blbr_jord8(v, al, dm)
            bl, br = yppm.bl_br_edges(bl, br, v, dya, al, dm)

            with horizontal(region[:, j_start + 1], region[:, j_end - 1]):
                bl, br = yppm.pert_ppm_standard_constraint_fcn(v, bl, br)

        # Zero corners
        with horizontal(
            region[i_start, j_start - 1 : j_start + 1],
            region[i_start, j_end : j_end + 2],
            region[i_end + 1, j_start - 1 : j_start + 1],
            region[i_end + 1, j_end : j_end + 2],
        ):
            bl = 0.0
            br = 0.0

        flux = _get_flux(v, courant, rdy, bl, br)


class YTP_V:
    def __init__(self, namelist):
        jord = spec.namelist.hord_mt
        if jord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert namelist.grid_type < 3

        grid = spec.grid
        origin = grid.compute_origin()
        domain = grid.domain_shape_compute(add=(1, 1, 0))
        self.dy = grid.dy
        self.dya = grid.dya
        self.rdy = grid.rdy
        ax_offsets = axis_offsets(grid, origin, domain)
        assert namelist.grid_type < 3

        self.stencil = FrozenStencil(
            _compute_stencil,
            externals={
                "jord": jord,
                "mord": jord,
                "xt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, c: FloatField, v: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in y-dir.

        Args:
        c (in): Courant number in flux form
        v (in): y-dir wind on Arakawa D-grid
        flux (out): Flux of kinetic energy
        """

        self.stencil(c, v, flux, self.dy, self.dya, self.rdy)
