import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
import pace.util.constants as constants
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import GridIndexing, StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.fv3core.stencils.delnflux import DelnFluxNoSG
from pace.fv3core.stencils.fvtp2d import FiniteVolumeTransport
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
    dp0: FloatFieldK,
    gk: FloatField,
    beta: FloatField,
    gamma: FloatField,
):
    """
    Computes constants used in cubic spline interpolation
    from cell center to interface levels.

    Args:
        dp0: target pressure on interface levels (in)
        gk: interpolation constant on mid levels (out)
        beta: interpolation constant on mid levels (out)
        gamma: interpolation constant on mid levels (out)
    """
    with computation(FORWARD):
        with interval(0, 1):
            gk = dp0[1] / dp0
            beta = gk * (gk + 0.5)
            gamma = (1.0 + gk * (gk + 1.5)) / beta
        with interval(1, -1):
            gk = dp0[-1] / dp0
            beta = 2.0 + 2.0 * gk - gamma[0, 0, -1]
            gamma = gk / beta


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
        damping_coefficients: DampingCoefficients,
        grid_data: GridData,
        grid_type: int,
        hord_tm: int,
        dp0: FloatFieldK,
        column_namelist,
    ):
        """
        Args:
            grid: fv3core grid object
            namelist: flattened fv3gfs namelist
            dp0: air pressure on interface levels, reference pressure
                can be used as an approximation
            column_namelist: dictionary of parameter columns
        """
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        grid_indexing = stencil_factory.grid_indexing
        self.grid_indexing = grid_indexing
        self._area = grid_data.area
        self._column_namelist = column_namelist
        if any(column_namelist["damp_vt"] <= 1e-5):
            raise NotImplementedError("damp <= 1e-5 in column_cols is untested")
        self._dp0 = dp0
        self._allocate_temporary_storages(
            grid_indexing, backend=stencil_factory.backend
        )
        self._initialize_interpolation_constants(
            stencil_factory=stencil_factory, grid_indexing=grid_indexing
        )

        self._interpolate_to_layer_interface = stencil_factory.from_origin_domain(
            cubic_spline_interpolation_from_layer_center_to_interfaces,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self.finite_volume_transport = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
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

    def _allocate_temporary_storages(self, grid_indexing: GridIndexing, backend: str):
        largest_possible_shape = grid_indexing.domain_full(add=(1, 1, 1))
        self._crx_interface = utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(0, -grid_indexing.n_halo, 0)),
            backend=backend,
        )
        self._cry_interface = utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(-grid_indexing.n_halo, 0, 0)),
            backend=backend,
        )
        self._x_area_flux_interface = utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(0, -grid_indexing.n_halo, 0)),
            backend=backend,
        )
        self._y_area_flux_interface = utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(-grid_indexing.n_halo, 0, 0)),
            backend=backend,
        )
        self._wk = utils.make_storage_from_shape(
            largest_possible_shape, grid_indexing.origin_full(), backend=backend
        )
        self._height_x_diffusive_flux = utils.make_storage_from_shape(
            largest_possible_shape, grid_indexing.origin_full(), backend=backend
        )
        self._height_y_diffusive_flux = utils.make_storage_from_shape(
            largest_possible_shape, grid_indexing.origin_full(), backend=backend
        )
        self._fx = utils.make_storage_from_shape(
            largest_possible_shape, grid_indexing.origin_full(), backend=backend
        )
        self._fy = utils.make_storage_from_shape(
            largest_possible_shape, grid_indexing.origin_full(), backend=backend
        )

    def _initialize_interpolation_constants(
        self, stencil_factory: StencilFactory, grid_indexing: GridIndexing
    ):
        # because stencils only work on 3D at the moment, need to compute in 3D
        # and then make these 1D
        gk_3d = utils.make_storage_from_shape(
            (1, 1, grid_indexing.domain[2] + 1),
            (0, 0, 0),
            backend=stencil_factory.backend,
        )
        gamma_3d = utils.make_storage_from_shape(
            (1, 1, grid_indexing.domain[2] + 1),
            (0, 0, 0),
            backend=stencil_factory.backend,
        )
        beta_3d = utils.make_storage_from_shape(
            (1, 1, grid_indexing.domain[2] + 1),
            (0, 0, 0),
            backend=stencil_factory.backend,
        )

        _cubic_spline_interpolation_constants = stencil_factory.from_origin_domain(
            cubic_spline_interpolation_constants,
            origin=(0, 0, 0),
            domain=(1, 1, grid_indexing.domain[2] + 1),
        )

        _cubic_spline_interpolation_constants(self._dp0, gk_3d, beta_3d, gamma_3d)
        self._gk = utils.make_storage_data(
            gk_3d[0, 0, :], gk_3d.shape[2:], (0,), backend=stencil_factory.backend
        )
        self._beta = utils.make_storage_data(
            beta_3d[0, 0, :], beta_3d.shape[2:], (0,), backend=stencil_factory.backend
        )
        self._gamma = utils.make_storage_data(
            gamma_3d[0, 0, :], gamma_3d.shape[2:], (0,), backend=stencil_factory.backend
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
