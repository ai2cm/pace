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
from fv3core.stencils import xppm, yppm
from fv3core.utils.grid import GridIndexing, axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def _get_flux(
    u: FloatField,
    courant: FloatField,
    rdx: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
):
    """
    Compute the x-dir flux of kinetic energy(?).

    Inputs:
        u: x-dir wind
        courant: Courant number in flux form
        rdx: 1.0 / dx
        bl: ???
        br: ???

    Returns:
        flux: Kinetic energy flux
    """
    # Could try merging this with xppm version.

    from __externals__ import iord

    b0 = bl + br
    cfl = courant * rdx[-1, 0] if courant > 0 else courant * rdx
    fx0 = xppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(iord < 8):
        tmp = xppm.get_tmp(bl, b0, br)
    else:
        tmp = 1.0
    return xppm.final_flux(courant, u, fx0, tmp)


def _xtp_u(
    courant: FloatField,
    u: FloatField,
    flux: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, iord, j_end, j_start

    with computation(PARALLEL), interval(...):

        if __INLINED(iord < 8):
            al = xppm.compute_al(u, dx)

            bl = al[0, 0, 0] - u[0, 0, 0]
            br = al[1, 0, 0] - u[0, 0, 0]

        else:
            dm = xppm.dm_iord8plus(u)
            al = xppm.al_iord8plus(u, dm)

            compile_assert(iord == 8)

            bl, br = xppm.blbr_iord8(u, al, dm)
            bl, br = xppm.bl_br_edges(bl, br, u, dxa, al, dm)

            with horizontal(region[i_start + 1, :], region[i_end - 1, :]):
                bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

        # Zero corners
        with horizontal(
            region[i_start - 1 : i_start + 1, j_start],
            region[i_start - 1 : i_start + 1, j_end + 1],
            region[i_end : i_end + 2, j_start],
            region[i_end : i_end + 2, j_end + 1],
        ):
            bl = 0.0
            br = 0.0

        flux = _get_flux(u, courant, rdx, bl, br)


class XTP_U:
    def __init__(
        self, grid_indexing: GridIndexing, dx, dxa, rdx, grid_type: int, iord: int
    ):
        if iord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self.dx = dx
        self.dxa = dxa
        self.rdx = rdx
        ax_offsets = axis_offsets(grid_indexing, origin, domain)
        self.stencil = FrozenStencil(
            _xtp_u,
            externals={
                "iord": iord,
                "mord": iord,
                "xt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, c: FloatField, u: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in x-dir.

        Args:
            c (in): Courant number in flux form
            u (in): x-dir wind on D-grid
            flux (out): Flux of kinetic energy
        """
        self.stencil(
            c,
            u,
            flux,
            self.dx,
            self.dxa,
            self.rdx,
        )
