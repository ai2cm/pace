from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.delnflux import DelnFlux
from fv3core.stencils.xppm import XPiecewiseParabolic
from fv3core.stencils.yppm import YPiecewiseParabolic
from fv3core.utils.grid import GridIndexing
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

    def __init__(
        self,
        grid_indexing: GridIndexing,
        dxa,
        dya,
        area,
        del6_u,
        del6_v,
        rarea,
        da_min,
        grid_type: int,
        hord,
        nord=None,
        damp_c=None,
    ):
        # use a shorter alias for grid_indexing here to avoid very verbose lines
        idx = grid_indexing
        self._area = area
        origin = idx.origin_compute()
        self._tmp_q_i = utils.make_storage_from_shape(idx.max_shape, origin)
        self._tmp_q_j = utils.make_storage_from_shape(idx.max_shape, origin)
        self._tmp_fx2 = utils.make_storage_from_shape(idx.max_shape, origin)
        self._tmp_fy2 = utils.make_storage_from_shape(idx.max_shape, origin)
        self._corner_tmp = utils.make_storage_from_shape(
            idx.max_shape, origin=idx.origin_full()
        )
        """Temporary field to use for corner computation in both x and y direction"""
        self._nord = nord
        self._damp_c = damp_c
        ord_outer = hord
        ord_inner = 8 if hord == 10 else hord
        self.stencil_q_i = FrozenStencil(
            q_i_stencil,
            origin=idx.origin_full(add=(0, 3, 0)),
            domain=idx.domain_full(add=(0, -3, 1)),
        )
        self.stencil_q_j = FrozenStencil(
            q_j_stencil,
            origin=idx.origin_full(add=(3, 0, 0)),
            domain=idx.domain_full(add=(-3, 0, 1)),
        )
        self.stencil_transport_flux = FrozenStencil(
            transport_flux_xy,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )
        if (self._nord is not None) and (self._damp_c is not None):
            self.delnflux: Optional[DelnFlux] = DelnFlux(
                grid_indexing, del6_u, del6_v, rarea, da_min, self._nord, self._damp_c
            )
        else:
            self.delnflux = None

        self.x_piecewise_parabolic_inner = XPiecewiseParabolic(
            grid_indexing=grid_indexing,
            dxa=dxa,
            grid_type=grid_type,
            iord=ord_inner,
            origin=idx.origin_compute(add=(0, -idx.n_halo, 0)),
            domain=idx.domain_compute(add=(1, 1 + 2 * idx.n_halo, 1)),
        )
        self.y_piecewise_parabolic_inner = YPiecewiseParabolic(
            grid_indexing=grid_indexing,
            dya=dya,
            grid_type=grid_type,
            jord=ord_inner,
            origin=idx.origin_compute(add=(-idx.n_halo, 0, 0)),
            domain=idx.domain_compute(add=(1 + 2 * idx.n_halo, 1, 1)),
        )
        self.x_piecewise_parabolic_outer = XPiecewiseParabolic(
            grid_indexing=grid_indexing,
            dxa=dxa,
            grid_type=grid_type,
            iord=ord_outer,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )
        self.y_piecewise_parabolic_outer = YPiecewiseParabolic(
            grid_indexing=grid_indexing,
            dya=dya,
            grid_type=grid_type,
            jord=ord_outer,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(1, 1, 1)),
        )

        self._copy_corners_x: corners.CopyCorners = corners.CopyCorners(
            "x", self._corner_tmp
        )
        """Stencil responsible for doing corners updates in x-direction."""

        self._copy_corners_y: corners.CopyCorners = corners.CopyCorners(
            "y", self._corner_tmp
        )
        """Stencil responsible for doing corners updates in y-direction."""

    def __call__(
        self,
        q,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        fx,
        fy,
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
            mass: ???
            mfx: ???
            mfy: ???
        """
        self._copy_corners_y(q)

        self.y_piecewise_parabolic_inner(q, cry, self._tmp_fy2)
        self.stencil_q_i(
            q,
            self._area,
            y_area_flux,
            self._tmp_fy2,
            self._tmp_q_i,
        )
        self.x_piecewise_parabolic_outer(self._tmp_q_i, crx, fx)

        self._copy_corners_x(q)

        self.x_piecewise_parabolic_inner(q, crx, self._tmp_fx2)
        self.stencil_q_j(
            q,
            self._area,
            x_area_flux,
            self._tmp_fx2,
            self._tmp_q_j,
        )
        self.y_piecewise_parabolic_outer(self._tmp_q_j, cry, fy)
        if mfx is not None and mfy is not None:
            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                mfx,
                mfy,
            )
            if (mass is not None) and self.delnflux is not None:
                self.delnflux(q, fx, fy, mass=mass)
        else:
            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                x_area_flux,
                y_area_flux,
            )
            if self.delnflux is not None:
                self.delnflux(q, fx, fy)
