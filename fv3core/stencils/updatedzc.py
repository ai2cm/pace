import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core.utils.global_constants as constants
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils import basic_operations
from fv3core.utils import corners, gt4py_utils
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


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
    fx = xfx * (gz_x[-1, 0, 0] if xfx > 0.0 else gz_x)
    fy = yfx * (gz_y[0, -1, 0] if yfx > 0.0 else gz_y)
    return fx, fy


def set_zero_2d(out_field: FloatFieldIJ):
    with computation(FORWARD):
        with interval(0, 1):
            out_field = 0.0  # in_field
        with interval(1, None):
            out_field = out_field


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
    with computation(PARALLEL):
        with interval(0, 1):
            xfx = p_weighted_average_top(ut, dp_ref)
            yfx = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            xfx = p_weighted_average_domain(ut, dp_ref)
            yfx = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            xfx = p_weighted_average_bottom(ut, dp_ref)
            yfx = p_weighted_average_bottom(vt, dp_ref)
    with computation(PARALLEL), interval(...):
        fx, fy = xy_flux(gz_x, gz_y, xfx, yfx)
        # TODO: check if below gz is ok, or if we need gz_y to pass this
        gz = (gz_y * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
            area + xfx - xfx[1, 0, 0] + yfx - yfx[0, 1, 0]
        )
    with computation(FORWARD), interval(-1, None):
        rdt = 1.0 / dt
        ws = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        gz_kp1 = gz[0, 0, 1] + DZ_MIN
        gz = gz if gz > gz_kp1 else gz_kp1


class UpdateGeopotentialHeightOnCGrid:
    def __init__(self, grid):
        self.grid = grid
        largest_possible_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._gz_in = gt4py_utils.make_storage_from_shape(
            largest_possible_shape,
            self.grid.compute_origin(add=(0, -self.grid.halo, 0)),
        )
        self._gz_x = gt4py_utils.make_storage_from_shape(
            largest_possible_shape,
            self.grid.compute_origin(add=(0, -self.grid.halo, 0)),
        )
        self._gz_y = gt4py_utils.make_storage_from_shape(
            largest_possible_shape,
            self.grid.compute_origin(add=(0, -self.grid.halo, 0)),
        )
        self._update_dz_c = FrozenStencil(
            update_dz_c,
            origin=(1, 1, 0),
            domain=(grid.nic + 3, grid.njc + 3, grid.npz + 1),
        )
        # TODO: convert to FrozenStencil when we have selective validation
        self._set_zero_2d = gtstencil(set_zero_2d)

    def __call__(
        self,
        dp_ref: FloatFieldK,
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
            gz: geopotential height (TODO: on cell mid levels or interfaces?)
            ws: surface vertical wind implied by horizontal motion over topography
            dt: timestep over which to evolve the geopotential height
        """
        basic_operations.copy_stencil(
            gz,
            self._gz_in,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        basic_operations.copy_stencil(
            gz,
            self._gz_x,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        basic_operations.copy_stencil(
            gz,
            self._gz_y,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )

        corners.fill_corners_2cells_x_stencil(
            self._gz_x,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        corners.fill_corners_2cells_y_stencil(
            self._gz_y,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )

        self._update_dz_c(
            dp_ref,
            zs,
            self.grid.area,
            ut,
            vt,
            gz,
            self._gz_x,
            self._gz_y,
            ws,
            dt=dt,
        )

        # should be able to combine these set_zero_2d with selective validation
        self._set_zero_2d(ws, origin=(1, 1, 0), domain=(1, self.grid.njc + 3, 1))
        self._set_zero_2d(ws, origin=(1, 1, 0), domain=(self.grid.nic + 3, 1, 1))
        # similarly should be able to combine these copies with selective validation
        basic_operations.copy_stencil(
            self._gz_in,
            gz,
            origin=(1, 1, 0),
            domain=(1, self.grid.njc + 3, self.grid.npz + 1),
        )
        basic_operations.copy_stencil(
            self._gz_in,
            gz,
            origin=(1, 1, 0),
            domain=(self.grid.nic + 3, 1, self.grid.npz + 1),
        )
