import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core.utils.global_constants as constants
from fv3core.decorators import FrozenStencil
from fv3core.utils import corners, gt4py_utils
from fv3core.utils.grid import axis_offsets
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
    def __init__(self, grid_indexing, area):
        self._area = area
        largest_possible_shape = grid_indexing.domain_full(add=(1, 1, 1))
        self._gz_x = gt4py_utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(0, -grid_indexing.n_halo, 0)),
        )
        self._gz_y = gt4py_utils.make_storage_from_shape(
            largest_possible_shape,
            grid_indexing.origin_compute(add=(0, -grid_indexing.n_halo, 0)),
        )
        full_origin = grid_indexing.origin_full()
        full_domain = grid_indexing.domain_full(add=(0, 0, 1))
        self._double_copy_stencil = FrozenStencil(
            double_copy,
            origin=full_origin,
            domain=full_domain,
        )

        ax_offsets = axis_offsets(grid_indexing, full_origin, full_domain)
        self._fill_corners_x_stencil = FrozenStencil(
            corners.fill_corners_2cells_x_stencil,
            externals=ax_offsets,
            origin=full_origin,
            domain=full_domain,
        )
        self._fill_corners_y_stencil = FrozenStencil(
            corners.fill_corners_2cells_y_stencil,
            externals=ax_offsets,
            origin=full_origin,
            domain=full_domain,
        )
        self._update_dz_c = FrozenStencil(
            update_dz_c,
            origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
            domain=grid_indexing.domain_compute(add=(2, 2, 1)),
        )

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
        # TODO: use a tmp variable inside the update_dz_c stencil instead of
        # _gz_x and _gz_y stencil to skip the copies and corner-fill stencils
        # once regions bug is fixed
        self._double_copy_stencil(gz, self._gz_x, self._gz_y)

        self._fill_corners_x_stencil(
            self._gz_x,
        )
        self._fill_corners_y_stencil(
            self._gz_y,
        )

        self._update_dz_c(
            dp_ref,
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
