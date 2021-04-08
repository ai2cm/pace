import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.global_constants as constants
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy, copy_stencil
from fv3core.utils import corners
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


@gtstencil()
def set_zero_2d(out_field: FloatFieldIJ):
    with computation(FORWARD):
        with interval(0, 1):
            out_field = 0.0  # in_field
        with interval(1, None):
            out_field = out_field


@gtstencil()
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


def compute(
    dp_ref: FloatFieldK,
    zs: FloatFieldIJ,
    ut: FloatField,
    vt: FloatField,
    gz: FloatField,
    ws: FloatFieldIJ,
    dt2: float,
):
    grid = spec.grid
    origin = (1, 1, 0)
    gz_in = copy(gz, origin=origin, cache_key="updatedzc_gz")
    gz_x = copy(gz, origin=origin, cache_key="updatedzc_gz_x")

    # corners.fill_corners_cells(gz_x, "x")
    corners.fill_corners_2cells_x_stencil(
        gz_x, origin=grid.full_origin(), domain=grid.domain_shape_full(add=(0, 0, 1))
    )
    gz_y = copy(gz_x, origin=origin, cache_key="updatedzc_gz_y")
    # corners.fill_corners_cells(gz_y, "y")
    corners.fill_corners_2cells_y_stencil(
        gz_y, origin=grid.full_origin(), domain=grid.domain_shape_full(add=(0, 0, 1))
    )
    update_dz_c(
        dp_ref,
        zs,
        grid.area,
        ut,
        vt,
        gz,
        gz_x,
        gz_y,
        ws,
        dt=dt2,
        origin=origin,
        domain=(grid.nic + 3, grid.njc + 3, grid.npz + 1),
    )

    set_zero_2d(ws, origin=(1, 1, 0), domain=(1, grid.njc + 3, 1))
    set_zero_2d(ws, origin=(1, 1, 0), domain=(grid.nic + 3, 1, 1))
    copy_stencil(gz_in, gz, origin=(1, 1, 0), domain=(1, grid.njc + 3, grid.npz + 1))
    copy_stencil(gz_in, gz, origin=(1, 1, 0), domain=(grid.nic + 3, 1, grid.npz + 1))
