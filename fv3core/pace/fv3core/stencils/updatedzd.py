from typing import Tuple

import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.util
import pace.util.constants as constants
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.fv3core.stencils.delnflux import DelnFluxNoSG
from pace.fv3core.stencils.fvtp2d import FiniteVolumeTransport
from pace.util import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from pace.util.grid import DampingCoefficients, GridData


DZ_MIN = constants.DZ_MIN


@gtscript.function
def _apply_height_advective_flux(
    height: FloatField,
    area: FloatFieldIJ,
    x_height_flux: FloatField,
    y_height_flux: FloatField,
    x_area_flux: FloatField,
    y_area_flux: FloatField,
):
    """
    Apply the computed fluxes of height and gridcell area to the height profile.

    A positive flux of area corresponds to convergence, which expands
    the layer thickness.

    Args:
        height: initial height
        area: gridcell area in m^2
        x_height_flux: area-weighted flux of height in x-direction,
            in units of g * m^3
        y_height_flux: area-weighted flux of height in y-direction,
            in units of g * m^3
        x_area_flux: flux of area in x-direction, in units of m^2
        y_area_flux: flux of area in y-direction, in units of m^2
    """
    # described in Putman and Lin 2007 equation 7
    # updated area is used because of implicit-in-time evaluation
    area_after_flux = (
        (area + x_area_flux - x_area_flux[1, 0, 0])
        + (area + y_area_flux - y_area_flux[0, 1, 0])
        - area
    )
    # final height is the original volume plus the fluxed volumes,
    # divided by the final area
    return (
        height * area
        + x_height_flux
        - x_height_flux[1, 0, 0]
        + y_height_flux
        - y_height_flux[0, 1, 0]
    ) / area_after_flux


def apply_height_fluxes(
    area: FloatFieldIJ,
    height: FloatField,
    fx: FloatField,
    fy: FloatField,
    x_area_flux: FloatField,
    y_area_flux: FloatField,
    gz_x_diffusive_flux: FloatField,
    gz_y_diffusive_flux: FloatField,
    surface_height: FloatFieldIJ,
    ws: FloatFieldIJ,
    dt: float,
):
    """
    Apply all computed fluxes to height profile.

    All vertically-resolved arguments are defined on the same grid
    (normally interface levels).

    Args:
        area (in): gridcell area in m^2
        height (inout): height profile on which to apply fluxes
        fx (in): area-weighted flux of height in x-direction,
            in units of g * m^3
        fy (in): area-weighted flux of height in y-direction,
            in units of g * m^3
        x_area_flux (in): flux of area in x-direction, in units of m^2
        y_area_flux (in): flux of area in y-direction, in units of m^2
        gz_x_diffusive_flux (in): diffusive flux of area-weighted height
            in x-direction
        gz_y_diffusive_flux (in): diffusive flux of area-weighted height
            in y-direction
        surface_height (in): surface height
        ws (out): vertical velocity of the lowest level (to keep it at the surface)
        dt (in): acoustic timestep (seconds)
    Grid variable inputs:
        area
    """
    with computation(PARALLEL), interval(...):
        height = (
            _apply_height_advective_flux(height, area, fx, fy, x_area_flux, y_area_flux)
            + (
                gz_x_diffusive_flux
                - gz_x_diffusive_flux[1, 0, 0]
                + gz_y_diffusive_flux
                - gz_y_diffusive_flux[0, 1, 0]
            )
            / area
        )

    with computation(BACKWARD):
        with interval(-1, None):
            ws = (surface_height - height) / dt
        with interval(0, -1):
            # ensure layer thickness exceeds minimum
            other = height[0, 0, 1] + DZ_MIN
            height = height if height > other else other


def cubic_spline_interpolation_constants(
    dp0: pace.util.Quantity, quantity_factory: pace.util.QuantityFactory
) -> Tuple[pace.util.Quantity, pace.util.Quantity, pace.util.Quantity]:
    """
    Computes constants used in cubic spline interpolation
    from cell center to interface levels.

    Args:
        dp0: reference pressure thickness on mid levels (in)

    Returns:
        gk: interpolation constant on mid levels
        beta: interpolation constant on mid levels
        gamma: interpolation constant on mid levels
    """
    gk = quantity_factory.zeros([Z_DIM], units="")
    beta = quantity_factory.zeros([Z_DIM], units="")
    gamma = quantity_factory.zeros([Z_DIM], units="")
    gk.view[0] = dp0.view[1] / dp0.view[0]
    beta.view[0] = gk.view[0] * (gk.view[0] + 0.5)
    gamma.view[0] = (1.0 + gk.view[0] * (gk.view[0] + 1.5)) / beta.view[0]
    gk.view[1:] = dp0.view[:-1] / dp0.view[1:]
    for i in range(1, beta.view[:].shape[0]):
        beta.view[i] = 2.0 + 2.0 * gk.view[i] - gamma.view[i - 1]
        gamma.view[i] = gk.view[i] / beta.view[i]
    return gk, beta, gamma


def cubic_spline_interpolation_from_layer_center_to_interfaces(
    q_center: FloatField,
    q_interface: FloatField,
    gk: FloatFieldK,
    beta: FloatFieldK,
    gamma: FloatFieldK,
) -> FloatField:
    """
    Interpolate a field from layer (vertical) centers to interfaces.

    Corresponds to edge_profile in nh_utils.F90 in the original Fortran code.

    Args:
        q_center (in): value on layer centers
        q_interface (out): value on layer interfaces
        gk (in): cubic spline interpolation constant
        beta (in): cubic spline interpolation constant
        gamma (in): cubic spline interpolation constant
    """
    # NOTE: We have not ported the uniform_grid True option as it is never called
    # that way in this model. We have also ignored limiter != 0 for the same reason.
    with computation(FORWARD):
        with interval(0, 1):
            xt1 = 2.0 * gk * (gk + 1.0)
            q_interface = (xt1 * q_center + q_center[0, 0, 1]) / beta
        with interval(1, -1):
            q_interface = (
                3.0 * (q_center[0, 0, -1] + gk * q_center) - q_interface[0, 0, -1]
            ) / beta
        with interval(-1, None):
            a_bot = 1.0 + gk[-1] * (gk[-1] + 1.5)
            xt1 = 2.0 * gk[-1] * (gk[-1] + 1.0)
            xt2 = gk[-1] * (gk[-1] + 0.5) - a_bot * gamma[-1]
            q_interface = (
                xt1 * q_center[0, 0, -1]
                + q_center[0, 0, -2]
                - a_bot * q_interface[0, 0, -1]
            ) / xt2
    with computation(BACKWARD), interval(0, -1):
        q_interface -= gamma * q_interface[0, 0, 1]


class UpdateHeightOnDGrid:
    """
    Fortran name is updatedzd.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        damping_coefficients: DampingCoefficients,
        grid_data: GridData,
        grid_type: int,
        hord_tm: int,
        column_namelist,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        grid_indexing = stencil_factory.grid_indexing
        self.grid_indexing = grid_indexing
        self._area = grid_data.area
        self._column_namelist = column_namelist
        if any(column_namelist["damp_vt"].view[:] <= 1e-5):
            raise NotImplementedError("damp <= 1e-5 in column_namelist is untested")
        self._dp_ref = grid_data.dp_ref
        self._allocate_temporary_storages(quantity_factory)
        self._gk, self._beta, self._gamma = cubic_spline_interpolation_constants(
            dp0=grid_data.dp_ref, quantity_factory=quantity_factory
        )

        self._interpolate_to_layer_interface = stencil_factory.from_origin_domain(
            cubic_spline_interpolation_from_layer_center_to_interfaces,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self.finite_volume_transport = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=grid_type,
            hord=hord_tm,
        )
        self.delnflux = DelnFluxNoSG(
            stencil_factory,
            damping_coefficients,
            grid_data.rarea,
            self._column_namelist["nord_v"],
            nk=grid_indexing.domain[2] + 1,
        )
        self._apply_height_fluxes = stencil_factory.from_origin_domain(
            apply_height_fluxes,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )

    def _allocate_temporary_storages(self, quantity_factory: pace.util.QuantityFactory):
        self._crx_interface = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_DIM, Z_INTERFACE_DIM], ""
        )
        self._cry_interface = quantity_factory.zeros(
            [X_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM], ""
        )
        self._x_area_flux_interface = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_DIM, Z_INTERFACE_DIM], "m^2"
        )
        self._y_area_flux_interface = quantity_factory.zeros(
            [X_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM], "m^2"
        )
        self._wk = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "unknown")
        self._height_x_diffusive_flux = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], "unknown"
        )
        self._height_y_diffusive_flux = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], "unknown"
        )
        self._fx = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_DIM, Z_INTERFACE_DIM], "unknown"
        )
        self._fy = quantity_factory.zeros(
            [X_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM], "unknown"
        )

    def __call__(
        self,
        surface_height: FloatFieldIJ,
        height: FloatField,
        courant_number_x: FloatField,
        courant_number_y: FloatField,
        x_area_flux: FloatField,
        y_area_flux: FloatField,
        ws: FloatFieldIJ,
        dt: float,
    ):
        """
        Advect height on D-grid.

        Height can be in any units, including geopotential units.

        Args:
            surface_height (in): height of surface
            height (inout): height defined on layer interfaces
            courant_number_x (in): Courant number in x-direction defined on cell centers
            courant_number_y (in): Courant number in y-direction defined on cell centers
            x_area_flux (in): Area flux in x-direction defined on cell centers
            y_area_flux (in): Area flux in y-direction defined on cell centers
            ws (out): lowest layer vertical velocity implied by horizontal motion
                over topography, in units of [height units] / second
            dt (in): timestep over which input fluxes have been computed, in seconds
        """
        self._interpolate_to_layer_interface(
            courant_number_x, self._crx_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            x_area_flux, self._x_area_flux_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            courant_number_y, self._cry_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            y_area_flux, self._y_area_flux_interface, self._gk, self._beta, self._gamma
        )

        # compute fluxes
        self.finite_volume_transport(
            height,
            self._crx_interface,
            self._cry_interface,
            self._x_area_flux_interface,
            self._y_area_flux_interface,
            self._fx,
            self._fy,
        )

        # TODO: in theory, we should check if damp_vt > 1e-5 for each k-level and
        # only compute for k-levels where this is true
        # compute diffusive component of fluxes
        self.delnflux(
            height,
            self._height_x_diffusive_flux,
            self._height_y_diffusive_flux,
            self._column_namelist["damp_vt"],
            self._wk,
        )
        self._apply_height_fluxes(
            self._area,
            height,
            self._fx,
            self._fy,
            self._x_area_flux_interface,
            self._y_area_flux_interface,
            self._height_x_diffusive_flux,
            self._height_y_diffusive_flux,
            surface_height,
            ws,
            dt,
        )
