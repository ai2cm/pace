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

from fv3core.decorators import FrozenStencil
from fv3core.stencils import yppm
from fv3core.utils.grid import GridData, GridIndexing, axis_offsets
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


def _ytp_v(
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
    def __init__(
        self,
        grid_indexing: GridIndexing,
        grid_data: GridData,
        grid_type: int,
        jord: int,
    ):
        if jord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self._dy = grid_data.dy
        self._dya = grid_data.dya
        self._rdy = grid_data.rdy
        ax_offsets = axis_offsets(grid_indexing, origin, domain)

        self.stencil = FrozenStencil(
            _ytp_v,
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

        self.stencil(c, v, flux, self._dy, self._dya, self._rdy)
