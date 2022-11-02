import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.util
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.stencils import corners
from pace.util import X_DIM, Y_DIM, Z_DIM


DZ_MIN = constants.DZ_MIN


@gtscript.function
def p_weighted_average_top(vel, dp0):
    # TODO: ratio is a constant, where should this be placed?
    ratio = dp0 / (dp0 + dp0[1])
    return vel + (vel - vel[0, 0, 1]) * ratio


@gtscript.function
def p_weighted_average_bottom(vel, dp0):
    ratio = dp0[-1] / (dp0[-2] + dp0[-1])
    return vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * ratio


@gtscript.function
def p_weighted_average_domain(vel, dp0):
    int_ratio = 1.0 / (dp0[-1] + dp0)
    return (dp0 * vel[0, 0, -1] + dp0[-1] * vel) * int_ratio


@gtscript.function
def xy_flux(gz_x, gz_y, xfx, yfx):
    """
    Compute first-order upwind fluxes of gz in x and y directions.

    Args:
        gz_x: gz with corners copied to perform derivatives in x-direction
        gz_y: gz with corners copied to perform derivatives in y-direction
        xfx (out): contravariant c-grid u-wind interpolated to layer interfaces,
            including metric terms to make it a "volume flux"
        yfx (out): contravariant c-grid v-wind interpolated to layer interfaces

    Returns:
        fx: first-order upwind x-flux of gz
        fy: first-order upwind y-flux of gz
    """
    fx = xfx * (gz_x[-1, 0, 0] if xfx > 0.0 else gz_x)
    fy = yfx * (gz_y[0, -1, 0] if yfx > 0.0 else gz_y)
    return fx, fy


def double_copy(q_in: FloatField, copy_1: FloatField, copy_2: FloatField):
    with computation(PARALLEL), interval(...):
        copy_1 = q_in
        copy_2 = q_in


def update_dz_c(
    dp_ref: FloatFieldK,
    zs: FloatFieldIJ,
    area: FloatFieldIJ,
    ut: FloatField,
    vt: FloatField,
    gz: FloatField,
    gz_x: FloatField,
    gz_y: FloatField,
    ws: FloatFieldIJ,
    *,
    dt: float,
):
    """
    Step dz forward on c-grid
    Eusures gz is monotonically increasing in z at the end
    Args:
        dp_ref:
        zs:
        area:
        ut:
        vt:
        gz:
        gz_x: gz with corners copied to perform derivatives in x-direction
        gz_y: gz with corners copied to perform derivatives in y-direction
        ws: lagrangian (parcel-following) surface vertical wind implied by
            lowest-level gz change note that a parcel moving horizontally
            across terrain will be moving in the vertical (eqn 5.5 in documentation)
        dt:
    """

    # there's some complexity due to gz being defined on interfaces
    # have to interpolate winds to layer interfaces first, using higher-order
    # cubic spline interpolation
    with computation(PARALLEL):
        with interval(0, 1):
            # TODO: inline some or all of these functions
            xfx = p_weighted_average_top(ut, dp_ref)
            yfx = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            xfx = p_weighted_average_domain(ut, dp_ref)
            yfx = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            xfx = p_weighted_average_bottom(ut, dp_ref)
            yfx = p_weighted_average_bottom(vt, dp_ref)
    # xfx/yfx are now ut/vt interpolated to layer interfaces
    with computation(PARALLEL), interval(...):
        fx, fy = xy_flux(gz_x, gz_y, xfx, yfx)
        # TODO: check if below gz is ok, or if we need gz_y to pass this
        # probably okay with gz, only compute domain
        gz = (gz * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
            area + xfx - xfx[1, 0, 0] + yfx - yfx[0, 1, 0]
        )
    with computation(FORWARD), interval(-1, None):
        rdt = 1.0 / dt
        ws = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        gz_kp1 = gz[0, 0, 1] + DZ_MIN
        gz = gz if gz > gz_kp1 else gz_kp1


class UpdateGeopotentialHeightOnCGrid:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        area: pace.util.Quantity,
        dp_ref: pace.util.Quantity,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._area = area
        # TODO: this is needed because GridData.dp_ref does not have access
        # to a QuantityFactory, we should add a way to perform operations on
        # Quantity and persist the QuantityFactory choices
        # e.g. by adding a quantity.factory
        # attribute, or by implementing basic math like slicing, addition, etc.
        # here it's needed to ensure we have a buffer point after the compute domain
        self._dp_ref = quantity_factory.zeros(dp_ref.dims, units=dp_ref.units)
        self._dp_ref.view[:] = dp_ref.view[:]
        self._gz_x = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m**2/s**2")
        self._gz_y = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m**2/s**2")
        full_origin = grid_indexing.origin_full()
        full_domain = grid_indexing.domain_full(add=(0, 0, 1))
        self._double_copy_stencil = stencil_factory.from_origin_domain(
            double_copy,
            origin=full_origin,
            domain=full_domain,
        )

        ax_offsets = grid_indexing.axis_offsets(full_origin, full_domain)
        self._fill_corners_x_stencil = stencil_factory.from_origin_domain(
            corners.fill_corners_2cells_x_stencil,
            externals=ax_offsets,
            origin=full_origin,
            domain=full_domain,
        )
        self._fill_corners_y_stencil = stencil_factory.from_origin_domain(
            corners.fill_corners_2cells_y_stencil,
            externals=ax_offsets,
            origin=full_origin,
            domain=full_domain,
        )
        self._update_dz_c = stencil_factory.from_origin_domain(
            update_dz_c,
            origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
            domain=grid_indexing.domain_compute(add=(2, 2, 1)),
        )

    def __call__(
        self,
        zs: FloatFieldIJ,
        ut: FloatField,
        vt: FloatField,
        gz: FloatField,
        ws: FloatFieldIJ,
        dt: float,
    ):
        """
        Args:
            dp_ref: layer thickness in Pa
            zs: surface height in m
            ut: horizontal wind (TODO: covariant or contravariant?)
            vt: horizontal wind (TODO: covariant or contravariant?)
            gz: geopotential height on model interfaces
            ws: surface vertical wind implied by horizontal motion over topography
            dt: timestep over which to evolve the geopotential height, in seconds
        """

        # TODO: is this advecting gz, and if so can we name it that?
        # Can we reduce duplication of advection logic with other stencils?

        # TODO: use a tmp variable inside the update_dz_c stencil instead of
        # _gz_x and _gz_y stencil to skip the copies and corner-fill stencils
        # once regions bug is fixed
        self._double_copy_stencil(gz, self._gz_x, self._gz_y)

        # TODO(eddied): We pass the same fields 2x to avoid GTC validation errors
        self._fill_corners_x_stencil(self._gz_x, self._gz_x)
        self._fill_corners_y_stencil(self._gz_y, self._gz_y)

        self._update_dz_c(
            self._dp_ref,
            zs,
            self._area,
            ut,
            vt,
            gz,
            self._gz_x,
            self._gz_y,
            ws,
            dt=dt,
        )
