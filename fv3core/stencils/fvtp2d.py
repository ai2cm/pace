import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import StencilWrapper
from fv3core.stencils import d_sw, delnflux
from fv3core.stencils.xppm import XPiecewiseParabolic
from fv3core.stencils.yppm import YPiecewiseParabolic
from fv3core.utils.typing import FloatField, FloatFieldIJ


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
    fy2: FloatField,
    q_i: FloatField,
):
    with computation(PARALLEL), interval(...):
        fyy = y_area_flux * fy2
        area_with_y_flux = apply_y_flux_divergence(area, y_area_flux)
        q_i = (q * area + fyy - fyy[0, 1, 0]) / area_with_y_flux


def q_j_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    x_area_flux: FloatField,
    fx2: FloatField,
    q_j: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = x_area_flux * fx2
        area_with_x_flux = apply_x_flux_divergence(area, x_area_flux)
        q_j = (q * area + fx1 - fx1[1, 0, 0]) / area_with_x_flux


@gtscript.function
def transport_flux(f, f2, mf):
    return 0.5 * (f + f2) * mf


def transport_flux_xy(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mfx: FloatField,
    mfy: FloatField,
):
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, :-1]):
            fx = transport_flux(fx, fx2, mfx)
        with horizontal(region[:-1, :]):
            fy = transport_flux(fy, fy2, mfy)


class FiniteVolumeTransport:
    """
    Equivalent of Fortran FV3 subroutine fv_tp_2d, done in 3 dimensions.
    Tested on serialized data with FvTp2d
    ONLY USE_SG=False compiler flag implements
    """

    def __init__(self, namelist, hord):
        self.grid = spec.grid
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self._tmp_q_i = utils.make_storage_from_shape(shape, origin)
        self._tmp_q_j = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy2 = utils.make_storage_from_shape(shape, origin)
        ord_outer = hord
        ord_inner = 8 if hord == 10 else hord
        self.stencil_q_i = StencilWrapper(
            q_i_stencil,
            origin=self.grid.full_origin(add=(0, 3, 0)),
            domain=self.grid.domain_shape_full(add=(0, -3, 1)),
        )
        self.stencil_q_j = StencilWrapper(
            q_j_stencil,
            origin=self.grid.full_origin(add=(3, 0, 0)),
            domain=self.grid.domain_shape_full(add=(-3, 0, 1)),
        )
        self.stencil_transport_flux = StencilWrapper(
            transport_flux_xy,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 1)),
        )
        self.x_piecewise_parabolic_inner = XPiecewiseParabolic(namelist, ord_inner)
        self.y_piecewise_parabolic_inner = YPiecewiseParabolic(namelist, ord_inner)
        self.x_piecewise_parabolic_outer = XPiecewiseParabolic(namelist, ord_outer)
        self.y_piecewise_parabolic_outer = YPiecewiseParabolic(namelist, ord_outer)

    def __call__(
        self,
        q,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        fx,
        fy,
        nord=None,
        damp_c=None,
        mass=None,
        mfx=None,
        mfy=None,
    ):
        """
        Calculate fluxes for horizontal finite volume transport.

        Args:
            q: scalar to be transported (in)
            crx: Courant number in x-direction
            cry: Courant number in y-direction
            x_area_flux: flux of area in x-direction, in units of m^2 (in)
            y_area_flux: flux of area in y-direction, in units of m^2 (in)
            fx: transport flux of q in x-direction (out)
            fy: transport flux of q in y-direction (out)
            nord: ???
            damp_c: ???
            mass: ???
            mfx: ???
            mfy: ???
        """
        grid = self.grid
        corners.copy_corners_y_stencil(
            q, origin=grid.full_origin(), domain=grid.domain_shape_full(add=(0, 0, 1))
        )
        self.y_piecewise_parabolic_inner(q, cry, self._tmp_fy2, grid.isd, grid.ied)
        self.stencil_q_i(
            q,
            grid.area,
            y_area_flux,
            self._tmp_fy2,
            self._tmp_q_i,
        )
        self.x_piecewise_parabolic_outer(self._tmp_q_i, crx, fx, grid.js, grid.je)
        corners.copy_corners_x_stencil(
            q, origin=grid.full_origin(), domain=grid.domain_shape_full(add=(0, 0, 1))
        )
        self.x_piecewise_parabolic_inner(q, crx, self._tmp_fx2, grid.jsd, grid.jed)
        self.stencil_q_j(
            q,
            grid.area,
            x_area_flux,
            self._tmp_fx2,
            self._tmp_q_j,
        )
        self.y_piecewise_parabolic_outer(self._tmp_q_j, cry, fy, grid.is_, grid.ie)
        if mfx is not None and mfy is not None:
            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                mfx,
                mfy,
            )
            if (mass is not None) and (nord is not None) and (damp_c is not None):
                for kstart, nk in d_sw.k_bounds():
                    delnflux.compute_delnflux_no_sg(
                        q, fx, fy, nord[kstart], damp_c[kstart], kstart, nk, mass=mass
                    )
        else:
            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                x_area_flux,
                y_area_flux,
            )
            if (nord is not None) and (damp_c is not None):
                for kstart, nk in d_sw.k_bounds():
                    delnflux.compute_delnflux_no_sg(
                        q, fx, fy, nord[kstart], damp_c[kstart], kstart, nk
                    )
