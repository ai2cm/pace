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
            the y-wind on cell corners
        q: scalar being transported, on y-centers
        fx1: flux of q in units of q, on y-interfaces
        mask: fx1 is multiplied by this before being applied
    """
    return q[0, -1, 0] + fx1 * mask if courant > 0.0 else q + fx1 * mask


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    """
    Args:
        courant: courant number, v * dt / dy (unitless)
        br: ???
        b0: br + bl
        bl: ???
    """
    if courant > 0.0:
        ret = (1.0 - courant) * (br[0, -1, 0] - courant * b0[0, -1, 0])
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

    if smt5[0, -1, 0] or smt5[0, 0, 0]:
        advection_mask = 1.0
    else:
        advection_mask = 0.0

    return advection_mask


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[0, 1, 0] - q[0, 0, 0]
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
def dm_jord8plus(q: FloatField):
    yt = 0.25 * (q[0, 1, 0] - q[0, -1, 0])
    dqr = max(max(q, q[0, -1, 0]), q[0, 1, 0]) - q
    dql = q - min(min(q, q[0, -1, 0]), q[0, 1, 0])
    return sign(min(min(abs(yt), dqr), dql), yt)


@gtscript.function
def al_jord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[0, -1, 0] + q) + 1.0 / 3.0 * (dm[0, -1, 0] - dm)


@gtscript.function
def blbr_jord8(q: FloatField, al: FloatField, dm: FloatField):
    yt = 2.0 * dm
    bl = -1.0 * sign(min(abs(yt), abs(al - q)), yt)
    br = sign(min(abs(yt), abs(al[0, 1, 0] - q)), yt)
    return bl, br


@gtscript.function
def yt_dya_edge_0_base(q, dya):
    return 0.5 * (
        ((2.0 * dya + dya[0, -1]) * q - dya * q[0, -1, 0]) / (dya[0, -1] + dya)
        + ((2.0 * dya[0, 1] + dya[0, 2]) * q[0, 1, 0] - dya[0, 1] * q[0, 2, 0])
        / (dya[0, 1] + dya[0, 2])
    )


@gtscript.function
def yt_dya_edge_1_base(q, dya):
    return 0.5 * (
        ((2.0 * dya[0, -1] + dya[0, -2]) * q[0, -1, 0] - dya[0, -1] * q[0, -2, 0])
        / (dya[0, -2] + dya[0, -1])
        + ((2.0 * dya + dya[0, 1]) * q - dya * q[0, 1, 0]) / (dya + dya[0, 1])
    )


@gtscript.function
def yt_dya_edge_0(q, dya):
    from __externals__ import yt_minmax

    yt = yt_dya_edge_0_base(q, dya)
    if __INLINED(yt_minmax):
        minq = min(min(min(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        maxq = max(max(max(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        yt = min(max(yt, minq), maxq)
    return yt


@gtscript.function
def yt_dya_edge_1(q, dya):
    from __externals__ import yt_minmax

    yt = yt_dya_edge_1_base(q, dya)
    if __INLINED(yt_minmax):
        minq = min(min(min(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        maxq = max(max(max(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        yt = min(max(yt, minq), maxq)
    return yt


@gtscript.function
def compute_al(q: FloatField, dya: FloatFieldIJ):
    """
    Interpolate q at interface.

    Inputs:
        q: transported scalar centered along the y-direction
        dya: dy on A-grid (?)

    Returns:
        q interpolated to y-interfaces
    """
    from __externals__ import j_end, j_start, jord

    compile_assert(jord < 8)

    al = ppm.p1 * (q[0, -1, 0] + q) + ppm.p2 * (q[0, -2, 0] + q[0, 1, 0])

    if __INLINED(jord < 0):
        compile_assert(False)
        al = max(al, 0.0)

    with horizontal(region[:, j_start - 1], region[:, j_end]):
        al = ppm.c1 * q[0, -2, 0] + ppm.c2 * q[0, -1, 0] + ppm.c3 * q
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        al = 0.5 * (
            ((2.0 * dya[0, -1] + dya[0, -2]) * q[0, -1, 0] - dya[0, -1] * q[0, -2, 0])
            / (dya[0, -2] + dya[0, -1])
            + ((2.0 * dya[0, 0] + dya[0, 1]) * q[0, 0, 0] - dya[0, 0] * q[0, 1, 0])
            / (dya[0, 0] + dya[0, 1])
        )
    with horizontal(region[:, j_start + 1], region[:, j_end + 2]):
        al = ppm.c3 * q[0, -1, 0] + ppm.c2 * q[0, 0, 0] + ppm.c1 * q[0, 1, 0]

    return al


@gtscript.function
def bl_br_edges(bl, br, q, dya, al, dm):
    from __externals__ import j_end, j_start

    # TODO(eddied): This temporary prevents race conditions in regions
    al_ip1 = al[0, 1, 0]

    with horizontal(region[:, j_start - 1]):
        # TODO(rheag) when possible
        # dm_left = dm_jord8plus(q[0, -1, 0])
        yt = 0.25 * (q - q[0, -2, 0])
        dqr = max(max(q[0, -1, 0], q[0, -2, 0]), q) - q[0, -1, 0]
        dql = q[0, -1, 0] - min(min(q[0, -1, 0], q[0, -2, 0]), q)
        dm_left = sign(min(min(abs(yt), dqr), dql), yt)
        yt_bl = ppm.s14 * dm_left + ppm.s11 * (q[0, -1, 0] - q) + q
        yt_br = yt_dya_edge_0(q, dya)

    with horizontal(region[:, j_start]):
        # TODO(rheag) when possible
        # dm_right = dm_jord8plus(q[0, 1, 0])
        yt = 0.25 * (q[0, 2, 0] - q)
        dqr = max(max(q[0, 1, 0], q), q[0, 2, 0]) - q[0, 1, 0]
        dql = q[0, 1, 0] - min(min(q[0, 1, 0], q), q[0, 2, 0])
        dm_right = sign(min(min(abs(yt), dqr), dql), yt)
        yt_bl = ppm.s14 * dm_left + ppm.s11 * (q[0, -1, 0] - q) + q
        yt_bl = yt_dya_edge_1(q, dya)
        yt_br = ppm.s15 * q + ppm.s11 * q[0, 1, 0] - ppm.s14 * dm_right

    with horizontal(region[:, j_start + 1]):
        yt_bl = ppm.s15 * q[0, -1, 0] + ppm.s11 * q - ppm.s14 * dm
        yt_br = al_ip1

    with horizontal(region[:, j_end - 1]):
        yt_bl = al
        yt_br = ppm.s15 * q[0, 1, 0] + ppm.s11 * q + ppm.s14 * dm

    with horizontal(region[:, j_end]):
        # TODO(rheag) when possible
        # dm_left_end = dm_jord8plus(q[0, -1, 0])
        yt = 0.25 * (q - q[0, -2, 0])
        dqr = max(max(q[0, -1, 0], q[0, -2, 0]), q) - q[0, -1, 0]
        dql = q[0, -1, 0] - min(min(q[0, -1, 0], q[0, -2, 0]), q)
        dm_left_end = sign(min(min(abs(yt), dqr), dql), yt)
        yt_bl = ppm.s15 * q + ppm.s11 * q[0, -1, 0] + ppm.s14 * dm_left_end
        yt_br = yt_dya_edge_0(q, dya)

    with horizontal(region[:, j_end + 1]):
        # TODO(rheag) when possible
        # dm_right_end = dm_jord8plus(q[0, 1, 0])
        yt = 0.25 * (q[0, 2, 0] - q)
        dqr = max(max(q[0, 1, 0], q), q[0, 2, 0]) - q[0, 1, 0]
        dql = q[0, 1, 0] - min(min(q[0, 1, 0], q), q[0, 2, 0])
        dm_right_end = sign(min(min(abs(yt), dqr), dql), yt)
        yt_bl = yt_dya_edge_1(q, dya)
        yt_br = ppm.s11 * (q[0, 1, 0] - q) - ppm.s14 * dm_right_end + q

    with horizontal(
        region[:, j_start - 1 : j_start + 2], region[:, j_end - 1 : j_end + 2]
    ):
        bl = yt_bl - q
        br = yt_br - q

    return bl, br


@gtscript.function
def compute_blbr_ord8plus(q: FloatField, dya: FloatFieldIJ):
    from __externals__ import j_end, j_start, jord

    dm = dm_jord8plus(q)
    al = al_jord8plus(q, dm)

    compile_assert(jord == 8)

    bl, br = blbr_jord8(q, al, dm)
    bl, br = bl_br_edges(bl, br, q, dya, al, dm)

    with horizontal(
        region[:, j_start - 1 : j_start + 2], region[:, j_end - 1 : j_end + 2]
    ):
        bl, br = ppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    return bl, br


def compute_y_flux(
    q: FloatField, courant: FloatField, dya: FloatFieldIJ, yflux: FloatField
):
    """
    Args:
        q (in):
        courant (in):
        dya (in):
        yflux (out):
    """
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dya)
            yflux = get_flux(q, courant, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dya)
            yflux = get_flux_ord8plus(q, courant, bl, br)


class YPiecewiseParabolic:
    """
    Fortran name is xppm
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        dya,
        grid_type: int,
        jord,
        origin: Index3D,
        domain: Index3D,
    ):
        # Arguments come from:
        # namelist.grid_type
        # grid.dya
        assert grid_type < 3
        self._dya = dya
        ax_offsets = stencil_factory.grid_indexing.axis_offsets(origin, domain)
        self._compute_flux_stencil = stencil_factory.from_origin_domain(
            func=compute_y_flux,
            externals={
                "jord": jord,
                "mord": abs(jord),
                "yt_minmax": True,
                "j_start": ax_offsets["j_start"],
                "j_end": ax_offsets["j_end"],
            },
            origin=origin,
            domain=domain,
        )

    def __call__(
        self,
        q_in: FloatField,
        c: FloatField,
        q_mean_advected_through_y_interface: FloatField,
    ):
        """
        Determine the mean value of q_in to be advected along y-interfaces.

        This is done by integrating a piecewise-parabolic svbgrid reconstruction
        of q_in along the y-direction over the segment of gridcell which
        will be advected.

        Multiplying this mean value by the area to be advected through the interface
        would give the flux of q through that interface.

        Args:
            q_in (in): scalar to be integrated
            c (in): Courant number (v*dt/dy) in y-direction defined on y-interfaces,
                indicates the fraction of the adjacent grid cell which will be
                advected through the interface in one timestep
            q_mean_advected_through_y_interface (out): defined on y-interfaces.
                mean value of scalar within the segment of gridcell to be advected
                through that interface in one timestep, in units of q_in
        """
        # in the Fortran version of this code, "x_advection" routines
        # were called "get_flux", while the routine which got the flux was called
        # fx1_fn. The final value was called yflux instead of q_out.
        self._compute_flux_stencil(
            q_in, c, self._dya, q_mean_advected_through_y_interface
        )
        # bl and br are "edge perturbation values" as in equation 4.1
        # of the FV3 documentation
