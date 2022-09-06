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

from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, Index3D
from pace.fv3core.stencils import ppm
from pace.fv3core.stencils.basic_operations import sign


@gtscript.function
def apply_flux(courant, q, fx1, mask):
    """
    Args:
        courant: any value whose sign is the same as the sign of
            the x-wind on cell corners
        q: scalar being transported, on x-centers
        fx1: flux of q in units of q, on x-interfaces
        mask: fx1 is multiplied by this before being applied
    """
    return q[-1, 0, 0] + fx1 * mask if courant > 0.0 else q + fx1 * mask


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    """
    Args:
        courant: courant number, u * dt / dx (unitless)
        br: ???
        b0: br + bl
        bl: ???
    """
    if courant > 0.0:
        ret = (1.0 - courant) * (br[-1, 0, 0] - courant * b0[-1, 0, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@gtscript.function
def get_advection_mask(bl, b0, br):
    from __externals__ import mord

    if __INLINED(mord == 5):
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[-1, 0, 0] or smt5[0, 0, 0]:
        advection_mask = 1.0
    else:
        advection_mask = 0.0

    return advection_mask


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[1, 0, 0] - q[0, 0, 0]
    b0 = bl + br

    advection_mask = get_advection_mask(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return apply_flux(courant, q, fx1, advection_mask)  # noqa


@gtscript.function
def get_flux_ord8plus(
    q: FloatField, courant: FloatField, bl: FloatField, br: FloatField
):
    b0 = bl + br
    fx1 = fx1_fn(courant, br, b0, bl)
    return apply_flux(courant, q, fx1, 1.0)


@gtscript.function
def dm_iord8plus(q: FloatField):
    xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
    dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
    dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@gtscript.function
def al_iord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[-1, 0, 0] + q) + 1.0 / 3.0 * (dm[-1, 0, 0] - dm)


@gtscript.function
def blbr_iord8(q: FloatField, al: FloatField, dm: FloatField):
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


@gtscript.function
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0] + dxa)
        + ((2.0 * dxa[1, 0] + dxa[2, 0]) * q[1, 0, 0] - dxa[1, 0] * q[2, 0, 0])
        / (dxa[1, 0] + dxa[2, 0])
    )


@gtscript.function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
        / (dxa[-2, 0] + dxa[-1, 0])
        + ((2.0 * dxa + dxa[1, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0])
    )


@gtscript.function
def xt_dxa_edge_0(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_0_base(q, dxa)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_1_base(q, dxa)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def compute_al(q: FloatField, dxa: FloatFieldIJ):
    """
    Interpolate q at interface.

    Inputs:
        q: transported scalar centered along the x-direction
        dxa: dx on A-grid (?)

    Returns:
        q interpolated to x-interfaces
    """
    from __externals__ import i_end, i_start, iord

    compile_assert(iord < 8)

    al = ppm.p1 * (q[-1, 0, 0] + q) + ppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])

    if __INLINED(iord < 0):
        compile_assert(False)
        al = max(al, 0.0)

    with horizontal(region[i_start - 1, :], region[i_end, :]):
        al = ppm.c1 * q[-2, 0, 0] + ppm.c2 * q[-1, 0, 0] + ppm.c3 * q
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        al = 0.5 * (
            ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
            / (dxa[-2, 0] + dxa[-1, 0])
            + ((2.0 * dxa[0, 0] + dxa[1, 0]) * q[0, 0, 0] - dxa[0, 0] * q[1, 0, 0])
            / (dxa[0, 0] + dxa[1, 0])
        )
    with horizontal(region[i_start + 1, :], region[i_end + 2, :]):
        al = ppm.c3 * q[-1, 0, 0] + ppm.c2 * q[0, 0, 0] + ppm.c1 * q[1, 0, 0]

    return al


@gtscript.function
def bl_br_edges(bl, br, q, dxa, al, dm):
    from __externals__ import i_end, i_start

    # TODO(eddied): This temporary prevents race conditions in regions
    al_ip1 = al[1, 0, 0]

    with horizontal(region[i_start - 1, :]):
        # TODO(rheag) when possible
        # dm_left = dm_iord8plus(q[-1, 0, 0])
        xt = 0.25 * (q - q[-2, 0, 0])
        dqr = max(max(q[-1, 0, 0], q[-2, 0, 0]), q) - q[-1, 0, 0]
        dql = q[-1, 0, 0] - min(min(q[-1, 0, 0], q[-2, 0, 0]), q)
        dm_left = sign(min(min(abs(xt), dqr), dql), xt)
        xt_bl = ppm.s14 * dm_left + ppm.s11 * (q[-1, 0, 0] - q) + q
        xt_br = xt_dxa_edge_0(q, dxa)

    with horizontal(region[i_start, :]):
        # TODO(rheag) when possible
        # dm_right = dm_iord8plus(q[1, 0, 0])
        xt = 0.25 * (q[2, 0, 0] - q)
        dqr = max(max(q[1, 0, 0], q), q[2, 0, 0]) - q[1, 0, 0]
        dql = q[1, 0, 0] - min(min(q[1, 0, 0], q), q[2, 0, 0])
        dm_right = sign(min(min(abs(xt), dqr), dql), xt)
        xt_bl = ppm.s14 * dm_left + ppm.s11 * (q[-1, 0, 0] - q) + q
        xt_bl = xt_dxa_edge_1(q, dxa)
        xt_br = ppm.s15 * q + ppm.s11 * q[1, 0, 0] - ppm.s14 * dm_right

    with horizontal(region[i_start + 1, :]):
        xt_bl = ppm.s15 * q[-1, 0, 0] + ppm.s11 * q - ppm.s14 * dm
        xt_br = al_ip1

    with horizontal(region[i_end - 1, :]):
        xt_bl = al
        xt_br = ppm.s15 * q[1, 0, 0] + ppm.s11 * q + ppm.s14 * dm

    with horizontal(region[i_end, :]):
        # TODO(rheag) when possible
        # dm_left_end = dm_iord8plus(q[-1, 0, 0])
        xt = 0.25 * (q - q[-2, 0, 0])
        dqr = max(max(q[-1, 0, 0], q[-2, 0, 0]), q) - q[-1, 0, 0]
        dql = q[-1, 0, 0] - min(min(q[-1, 0, 0], q[-2, 0, 0]), q)
        dm_left_end = sign(min(min(abs(xt), dqr), dql), xt)
        xt_bl = ppm.s15 * q + ppm.s11 * q[-1, 0, 0] + ppm.s14 * dm_left_end
        xt_br = xt_dxa_edge_0(q, dxa)

    with horizontal(region[i_end + 1, :]):
        # TODO(rheag) when possible
        # dm_right_end = dm_iord8plus(q[1, 0, 0])
        xt = 0.25 * (q[2, 0, 0] - q)
        dqr = max(max(q[1, 0, 0], q), q[2, 0, 0]) - q[1, 0, 0]
        dql = q[1, 0, 0] - min(min(q[1, 0, 0], q), q[2, 0, 0])
        dm_right_end = sign(min(min(abs(xt), dqr), dql), xt)
        xt_bl = xt_dxa_edge_1(q, dxa)
        xt_br = ppm.s11 * (q[1, 0, 0] - q) - ppm.s14 * dm_right_end + q

    with horizontal(
        region[i_start - 1 : i_start + 2, :], region[i_end - 1 : i_end + 2, :]
    ):
        bl = xt_bl - q
        br = xt_br - q

    return bl, br


@gtscript.function
def compute_blbr_ord8plus(q: FloatField, dxa: FloatFieldIJ):
    from __externals__ import i_end, i_start, iord

    dm = dm_iord8plus(q)
    al = al_iord8plus(q, dm)

    compile_assert(iord == 8)

    bl, br = blbr_iord8(q, al, dm)
    bl, br = bl_br_edges(bl, br, q, dxa, al, dm)

    with horizontal(
        region[i_start - 1 : i_start + 2, :], region[i_end - 1 : i_end + 2, :]
    ):
        bl, br = ppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    return bl, br


def compute_x_flux(
    q: FloatField, courant: FloatField, dxa: FloatFieldIJ, xflux: FloatField
):
    """
    Args:
        q (in):
        courant (in):
        dxa (in):
        xflux (out):
    """
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dxa)
            xflux = get_flux(q, courant, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dxa)
            xflux = get_flux_ord8plus(q, courant, bl, br)


class XPiecewiseParabolic:
    """
    Fortran name is xppm
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        dxa,
        grid_type: int,
        iord,
        origin: Index3D,
        domain: Index3D,
    ):
        # Arguments come from:
        # namelist.grid_type
        # grid.dxa
        assert grid_type < 3
        self._dxa = dxa
        ax_offsets = stencil_factory.grid_indexing.axis_offsets(origin, domain)
        self._compute_flux_stencil = stencil_factory.from_origin_domain(
            func=compute_x_flux,
            externals={
                "iord": iord,
                "mord": abs(iord),
                "xt_minmax": True,
                "i_start": ax_offsets["i_start"],
                "i_end": ax_offsets["i_end"],
            },
            origin=origin,
            domain=domain,
        )

    def __call__(
        self,
        q_in: FloatField,
        c: FloatField,
        q_mean_advected_through_x_interface: FloatField,
    ):
        """
        Determine the mean value of q_in to be advected along x-interfaces.

        This is done by integrating a piecewise-parabolic subgrid reconstruction
        of q_in along the x-direction over the segment of gridcell which
        will be advected.

        Multiplying this mean value by the area to be advected through the interface
        would give the flux of q through that interface.

        Args:
            q_in (in): scalar to be integrated
            c (in): Courant number (u*dt/dx) in x-direction defined on x-interfaces,
                indicates the fraction of the adjacent grid cell which will be
                advected through the interface in one timestep
            q_mean_advected_through_x_interface (out): defined on x-interfaces.
                mean value of scalar within the segment of gridcell to be advected
                through that interface in one timestep, in units of q_in
        """
        # in the Fortran version of this code, "x_advection" routines
        # were called "get_flux", while the routine which got the flux was called
        # fx1_fn. The final value was called xflux instead of q_out.
        self._compute_flux_stencil(
            q_in, c, self._dxa, q_mean_advected_through_x_interface
        )
        # bl and br are "edge perturbation values" as in equation 4.1
        # of the FV3 documentation
