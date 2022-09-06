from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import pace.dsl.gt4py_utils as utils
import pace.stencils.corners as corners
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.delnflux import DelnFlux
from pace.fv3core.stencils.xppm import XPiecewiseParabolic
from pace.fv3core.stencils.yppm import YPiecewiseParabolic
from pace.util.grid import DampingCoefficients, GridData


@gtscript.function
def apply_x_flux_divergence(q: FloatField, q_x_flux: FloatField) -> FloatField:
    """
    Update a scalar q according to its flux in the x direction.
    """
    return q + q_x_flux - q_x_flux[1, 0, 0]


@gtscript.function
def apply_y_flux_divergence(q: FloatField, q_y_flux: FloatField) -> FloatField:
    """
    Update a scalar q according to its flux in the x direction.
    """
    return q + q_y_flux - q_y_flux[0, 1, 0]


def q_i_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    y_area_flux: FloatField,
    q_advected_along_y: FloatField,
    q_i: FloatField,
):
    """
    Args:
        q (in):
        area (in):
        y_area_flux (in):
        q_advected_along_y (in):
        q_i (out):
    """
    with computation(PARALLEL), interval(...):
        fyy = y_area_flux * q_advected_along_y
        # note the units of area cancel out, because area is present in all
        # terms in the numerator and denominator of q_i
        # corresponds to FV3 documentation eq 4.18, q_i = f(q)
        q_i = (q * area + fyy - fyy[0, 1, 0]) / (
            area + y_area_flux - y_area_flux[0, 1, 0]
        )


def q_j_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    x_area_flux: FloatField,
    fx2: FloatField,
    q_j: FloatField,
):
    """
    Args:
        q (in):
        area (in):
        x_area_flux (in):
        fx2 (in):
        q_j (out):
    """
    with computation(PARALLEL), interval(...):
        fx1 = x_area_flux * fx2
        area_with_x_flux = apply_x_flux_divergence(area, x_area_flux)
        q_j = (q * area + fx1 - fx1[1, 0, 0]) / area_with_x_flux


def final_fluxes(
    q_advected_y_x_advected_mean: FloatField,
    q_x_advected_mean: FloatField,
    q_advected_x_y_advected_mean: FloatField,
    q_y_advected_mean: FloatField,
    x_unit_flux: FloatField,
    y_unit_flux: FloatField,
    x_flux: FloatField,
    y_flux: FloatField,
):
    """
    Compute final x and y fluxes of q from different numerical representations.

    Corresponds roughly to eq. 4.17 of FV3 documentation, except that the flux
    is in units of q rather than in units of q per interface area per time.
    This corresponds to eq 4.17 with both sides multiplied by
    e.g. x_unit_flux / u^* (similarly for y/v).

    Combining the advection operators in this way is done to cancel leading-order
    numerical splitting error.

    Args:
        q_advected_y_x_advected_mean (in):
        q_x_advected_mean (in):
        q_advected_x_y_advected_mean (in):
        q_y_advected_mean (in):
        x_unit_flux (in):
        y_unit_flux (in):
        x_flux (out):
        y_flux (out):
    """
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, :-1]):
            x_flux = (
                0.5 * (q_advected_y_x_advected_mean + q_x_advected_mean) * x_unit_flux
            )
        with horizontal(region[:-1, :]):
            y_flux = (
                0.5 * (q_advected_x_y_advected_mean + q_y_advected_mean) * y_unit_flux
            )


class FiniteVolumeTransport:
    """
    Equivalent of Fortran FV3 subroutine fv_tp_2d, done in 3 dimensions.
    Tested on serialized data with FvTp2d
    ONLY USE_SG=False compiler flag implements
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        grid_type: int,
        hord,
        nord=None,
        damp_c=None,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )

        # use a shorter alias for grid_indexing here to avoid very verbose lines
        idx = stencil_factory.grid_indexing
        self._area = grid_data.area
        origin = idx.origin_compute()

        def make_storage():
            return utils.make_storage_from_shape(
                idx.max_shape,
                origin=origin,
                backend=stencil_factory.backend,
            )

        self._q_advected_y = make_storage()
        self._q_advected_x = make_storage()
        self._q_x_advected_mean = make_storage()
        self._q_y_advected_mean = make_storage()
        self._q_advected_x_y_advected_mean = make_storage()
        self._q_advected_y_x_advected_mean = make_storage()
        self._nord = nord
        self._damp_c = damp_c
        ord_outer = hord
        ord_inner = 8 if hord == 10 else hord
        if (self._nord is not None) and (self._damp_c is not None):
            # [DaCe] Use _do_delnflux instead of a None function
            # to have DaCe parsing working
            self._do_delnflux = True
            self.delnflux: Optional[DelnFlux] = DelnFlux(
                stencil_factory=stencil_factory,
                damping_coefficients=damping_coefficients,
                rarea=grid_data.rarea,
                nord=self._nord,
                damp_c=self._damp_c,
            )
        else:
            # This triggers dace parsing error:
            # self.delnflux = None
            self._do_delnflux = False

        self._copy_corners_y: corners.CopyCorners = corners.CopyCorners(
            "y", stencil_factory
        )
        self.y_piecewise_parabolic_inner = YPiecewiseParabolic(
            stencil_factory=stencil_factory,
            dya=grid_data.dya,
            grid_type=grid_type,
            jord=ord_inner,
            origin=idx.origin_compute(add=(-idx.n_halo, 0, 0)),
            domain=idx.domain_compute(add=(1 + 2 * idx.n_halo, 1, 1)),
        )
        self.q_i_stencil = stencil_factory.from_origin_domain(
            q_i_stencil,
            origin=idx.origin_full(add=(0, 3, 0)),
            domain=idx.domain_full(add=(0, -3, 1)),
        )
        self.x_piecewise_parabolic_outer = XPiecewiseParabolic(
            stencil_factory=stencil_factory,
            dxa=grid_data.dxa,
            grid_type=grid_type,
            iord=ord_outer,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )

        self._copy_corners_x: corners.CopyCorners = corners.CopyCorners(
            "x", stencil_factory
        )
        self.x_piecewise_parabolic_inner = XPiecewiseParabolic(
            stencil_factory=stencil_factory,
            dxa=grid_data.dxa,
            grid_type=grid_type,
            iord=ord_inner,
            origin=idx.origin_compute(add=(0, -idx.n_halo, 0)),
            domain=idx.domain_compute(add=(1, 1 + 2 * idx.n_halo, 1)),
        )
        self.q_j_stencil = stencil_factory.from_origin_domain(
            q_j_stencil,
            origin=idx.origin_full(add=(3, 0, 0)),
            domain=idx.domain_full(add=(-3, 0, 1)),
        )
        self.y_piecewise_parabolic_outer = YPiecewiseParabolic(
            stencil_factory=stencil_factory,
            dya=grid_data.dya,
            grid_type=grid_type,
            jord=ord_outer,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )
        self.stencil_transport_flux = stencil_factory.from_origin_domain(
            final_fluxes,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )

    def __call__(
        self,
        q,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        q_x_flux,
        q_y_flux,
        x_mass_flux=None,
        y_mass_flux=None,
        mass=None,
    ):
        """
        Calculate fluxes for horizontal finite volume transport.

        Defined in Putman and Lin 2007 (PL07). Corresponds to equation 4.17
        in the FV3 documentation.

        Divergence terms are handled by advecting the weighting used in
        the units of the scalar, and dividing by its divergence. For example,
        temperature (pt in the Fortran) and tracers are mass weighted, so
        the final tendency is
        e.g. (convergence of tracer) / (convergence of gridcell mass). This
        is described in eq 17 of PL07. pressure thickness and vorticity
        by contrast are area weighted.

        Args:
            q (in): scalar to be transported
            crx (in): Courant number in x-direction
            cry (in): Courant number in y-direction
            x_area_flux (in): flux of area in x-direction, in units of m^2
            y_area_flux (in): flux of area in y-direction, in units of m^2
            q_x_flux (out): transport flux of q in x-direction in units q * m^2,
                corresponding to X in eq 4.17 of FV3 documentation
            q_y_flux (out): transport flux of q in y-direction in units q * m^2,
                corresponding to Y in eq 4.17 of FV3 documentation
            x_mass_flux (in): mass flux in x-direction,
                corresponds to F(rho^* = 1) in PL07 eq 17, if not given
                then q is assumed to have per-area units
            y_mass_flux (in): mass flux in x-direction,
                corresponds to G(rho^* = 1) in PL07 eq 18, if not given
                then q is assumed to have per-area units
            mass (in): ??? passed along to damping code, if scalar is per-mass
                (as opposed to per-area) then this must be provided for
                damping to be correct
        """
        # [DaCe] dace.frontend.python.common.DaceSyntaxError: Keyword "Raise" disallowed
        # if (
        #     self.delnflux is not None
        #     and mass is None
        #     and (x_mass_flux is not None or y_mass_flux is not None)
        # ):
        #     raise ValueError(
        #         "when damping is enabled, mass must be given if mass flux is given"
        #     )
        if x_mass_flux is None:
            x_unit_flux = x_area_flux
        else:
            x_unit_flux = x_mass_flux
        if y_mass_flux is None:
            y_unit_flux = y_area_flux
        else:
            y_unit_flux = y_mass_flux

        # TODO: consider whether to refactor xppm/yppm to output fluxes by also taking
        # y_area_flux as an input (flux = area_flux * advected_mean), since a flux is
        # easier to understand than the current output. This would be like merging
        # yppm with q_i_stencil and xppm with q_j_stencil.
        self._copy_corners_y(q)
        self.y_piecewise_parabolic_inner(q, cry, self._q_y_advected_mean)
        # q_y_advected_mean is 1/Delta_area * curly-F, where curly-F is defined in
        # equation 4.3 of the FV3 documentation and Delta_area is the advected area
        # (y_area_flux)
        self.q_i_stencil(
            q,
            self._area,
            y_area_flux,
            self._q_y_advected_mean,
            self._q_advected_y,
        )  # q_advected_y out is f(q) in eq 4.18 of FV3 documentation
        self.x_piecewise_parabolic_outer(
            self._q_advected_y, crx, self._q_advected_y_x_advected_mean
        )
        # q_advected_y_x_advected_mean is now rho^n + F(rho^y) in PL07 eq 16

        self._copy_corners_x(q)
        # similarly below for x<->y
        self.x_piecewise_parabolic_inner(q, crx, self._q_x_advected_mean)
        self.q_j_stencil(
            q,
            self._area,
            x_area_flux,
            self._q_x_advected_mean,
            self._q_advected_x,
        )
        self.y_piecewise_parabolic_outer(
            self._q_advected_x, cry, self._q_advected_x_y_advected_mean
        )

        self.stencil_transport_flux(
            self._q_advected_y_x_advected_mean,
            self._q_x_advected_mean,
            self._q_advected_x_y_advected_mean,
            self._q_y_advected_mean,
            x_unit_flux,
            y_unit_flux,
            q_x_flux,
            q_y_flux,
        )
        if self._do_delnflux:
            self.delnflux(q, q_x_flux, q_y_flux, mass=mass)
