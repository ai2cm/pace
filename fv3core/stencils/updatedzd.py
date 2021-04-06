import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.delnflux as delnflux
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.stencils.fxadv import ra_x_func, ra_y_func
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


DZ_MIN = constants.DZ_MIN

# TODO merge with fxadv
@gtscript.function
def ra_func(
    area: FloatFieldIJ,
    xfx_adv: FloatField,
    yfx_adv: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, :]):
        ra_x = ra_x_func(area, xfx_adv)
    with horizontal(region[:, local_js : local_je + 2]):
        ra_y = ra_y_func(area, yfx_adv)
    return ra_x, ra_y


@gtstencil()
def ra_stencil_update(
    area: FloatFieldIJ,
    xfx_adv: FloatField,
    ra_x: FloatField,
    yfx_adv: FloatField,
    ra_y: FloatField,
):
    """Updates 'ra' fields.
    Args:
       xfx_adv: Finite volume flux form operator in x direction (in)
       yfx_adv: Finite volume flux form operator in y direction (in)
       ra_x: Area increased in the x direction due to flux divergence (inout)
       ra_y: Area increased in the y direction due to flux divergence (inout)
    Grid input vars:
      area
    """
    with computation(PARALLEL), interval(...):
        ra_x, ra_y = ra_func(area, xfx_adv, yfx_adv, ra_x, ra_y)


@gtscript.function
def zh_base(
    z2: FloatField,
    area: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
):
    return (z2 * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (ra_x + ra_y - area)


@gtstencil()
def zh_damp(
    area: FloatFieldIJ,
    z2: FloatField,
    fx: FloatField,
    fy: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
    fx2: FloatField,
    fy2: FloatField,
    rarea: FloatFieldIJ,
    zh: FloatField,
    zs: FloatFieldIJ,
    ws: FloatFieldIJ,
    dt: float,
):
    """Update geopotential height due to area average flux divergence
    Args:
       z2: zh that has been advected forward in time (in)
       fx: Flux in the x direction that transported z2 (in)
       fy: Flux in the y direction that transported z2(in)
       ra_x: Area increased in the x direction due to flux divergence (in)
       ra_y: Area increased in the y direction due to flux divergence (in)
       fx2: diffusive flux in the x-direction (in)
       fy2: diffusive flux in the y-direction (in)
       zh: geopotential height (out)
       zs: surface geopotential height (in)
       ws: vertical velocity of the lowest level (to keep it at the surface) (out)
       dt: acoustic timestep (seconds) (in)
    Grid variable inputs:
       area
      rarea
    """
    with computation(PARALLEL), interval(...):
        zhbase = zh_base(z2, area, fx, fy, ra_x, ra_y)
        zh = zhbase + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
    with computation(BACKWARD):
        with interval(-1, None):
            ws = (zs - zh) * 1.0 / dt
        with interval(0, -1):
            other = zh[0, 0, 1] + DZ_MIN
            zh = zh if zh > other else other


@gtstencil()
def zh_stencil(
    area: FloatFieldIJ,
    zh: FloatField,
    fx: FloatField,
    fy: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
):
    with computation(PARALLEL), interval(...):
        zh = zh_base(zh, area, fx, fy, ra_x, ra_y)


@gtscript.function
def edge_profile_top(
    dp0: FloatFieldK,
    q1x: FloatField,
    q2x: FloatField,
    qe1x: FloatField,
    qe2x: FloatField,
    q1y: FloatField,
    q2y: FloatField,
    qe1y: FloatField,
    qe2y: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    g0 = dp0[1] / dp0
    xt1 = 2.0 * g0 * (g0 + 1.0)
    bet = g0 * (g0 + 0.5)
    gam = (1.0 + g0 * (g0 + 1.5)) / bet

    with horizontal(region[local_is : local_ie + 2, :]):
        qe1x = (xt1 * q1x + q1x[0, 0, 1]) / bet
        qe2x = (xt1 * q2x + q2x[0, 0, 1]) / bet
    with horizontal(region[:, local_js : local_je + 2]):
        qe1y = (xt1 * q1y + q1y[0, 0, 1]) / bet
        qe2y = (xt1 * q2y + q2y[0, 0, 1]) / bet

    return qe1x, qe2x, qe1y, qe2y, gam


@gtscript.function
def edge_profile_reverse(
    qe1x: FloatField,
    qe2x: FloatField,
    qe1y: FloatField,
    qe2y: FloatField,
    gam: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, :]):
        qe1x -= gam * qe1x[0, 0, 1]
        qe2x -= gam * qe2x[0, 0, 1]
    with horizontal(region[:, local_js : local_je + 2]):
        qe1y -= gam * qe1y[0, 0, 1]
        qe2y -= gam * qe2y[0, 0, 1]

    return qe1x, qe2x, qe1y, qe2y


# NOTE: We have not ported the uniform_grid True option as it is never called
# that way in this model. We have also ignored limite != 0 for the same reason.
@gtstencil()
def edge_profile_stencil(
    q1x: FloatField,
    q2x: FloatField,
    qe1x: FloatField,
    qe2x: FloatField,
    q1y: FloatField,
    q2y: FloatField,
    qe1y: FloatField,
    qe2y: FloatField,
    dp0: FloatFieldK,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(FORWARD):
        with interval(0, 1):
            qe1x, qe2x, qe1y, qe2y, gam = edge_profile_top(
                dp0, q1x, q2x, qe1x, qe2x, q1y, q2y, qe1y, qe2y
            )
        with interval(1, -1):
            gk = dp0[-1] / dp0
            bet = 2.0 + 2.0 * gk - gam[0, 0, -1]
            gam = gk / bet
            with horizontal(region[local_is : local_ie + 2, :]):
                qe1x = (3.0 * (q1x[0, 0, -1] + gk * q1x) - qe1x[0, 0, -1]) / bet
                qe2x = (3.0 * (q2x[0, 0, -1] + gk * q2x) - qe2x[0, 0, -1]) / bet
            with horizontal(region[:, local_js : local_je + 2]):
                qe1y = (3.0 * (q1y[0, 0, -1] + gk * q1y) - qe1y[0, 0, -1]) / bet
                qe2y = (3.0 * (q2y[0, 0, -1] + gk * q2y) - qe2y[0, 0, -1]) / bet
        with interval(-1, None):
            a_bot = 1.0 + gk[0, 0, -1] * (gk[0, 0, -1] + 1.5)
            xt1 = 2.0 * gk[0, 0, -1] * (gk[0, 0, -1] + 1.0)
            xt2 = gk[0, 0, -1] * (gk[0, 0, -1] + 0.5) - a_bot * gam[0, 0, -1]
            with horizontal(region[local_is : local_ie + 2, :]):
                qe1x = (
                    xt1 * q1x[0, 0, -1] + q1x[0, 0, -2] - a_bot * qe1x[0, 0, -1]
                ) / xt2
                qe2x = (
                    xt1 * q2x[0, 0, -1] + q2x[0, 0, -2] - a_bot * qe2x[0, 0, -1]
                ) / xt2
            with horizontal(region[:, local_js : local_je + 2]):
                qe1y = (
                    xt1 * q1y[0, 0, -1] + q1y[0, 0, -2] - a_bot * qe1y[0, 0, -1]
                ) / xt2
                qe2y = (
                    xt1 * q2y[0, 0, -1] + q2y[0, 0, -2] - a_bot * qe2y[0, 0, -1]
                ) / xt2
    with computation(BACKWARD), interval(0, -1):
        qe1x, qe2x, qe1y, qe2y = edge_profile_reverse(qe1x, qe2x, qe1y, qe2y, gam)


def compute(
    dp0: FloatFieldK,
    zs: FloatFieldIJ,
    zh: FloatField,
    crx: FloatField,
    cry: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    wsd: FloatFieldIJ,
    dt: float,
):
    grid = spec.grid
    halo = grid.halo

    crx_adv = utils.make_storage_from_shape(
        crx.shape, grid.compute_origin(add=(0, -halo, 0)), cache_key="updatedzd_crx_adv"
    )
    cry_adv = utils.make_storage_from_shape(
        cry.shape, grid.compute_origin(add=(-halo, 0, 0)), cache_key="updatedzd_cry_adv"
    )
    xfx_adv = utils.make_storage_from_shape(
        xfx.shape, grid.compute_origin(add=(0, -halo, 0)), cache_key="updatedzd_xfx_adv"
    )
    yfx_adv = utils.make_storage_from_shape(
        yfx.shape, grid.compute_origin(add=(-halo, 0, 0)), cache_key="updatedzd_yfx_adv"
    )
    ra_x = utils.make_storage_from_shape(
        crx.shape,
        grid.compute_origin(add=(0, -halo, 0)),
        cache_key="updatedzd_ra_x",
    )
    ra_y = utils.make_storage_from_shape(
        cry.shape,
        grid.compute_origin(add=(-halo, 0, 0)),
        cache_key="updatedzd_ra_y",
    )

    edge_profile_stencil(
        crx,
        xfx,
        crx_adv,
        xfx_adv,
        cry,
        yfx,
        cry_adv,
        yfx_adv,
        dp0,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )
    ra_stencil_update(
        grid.area,
        xfx_adv,
        ra_x,
        yfx_adv,
        ra_y,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )

    grid = spec.grid

    wk = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_wk"
    )
    fx2 = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_fx2"
    )
    fy2 = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_fy2"
    )
    fx = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_fx"
    )
    fy = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_fy"
    )
    z2 = utils.make_storage_from_shape(
        zh.shape, grid.full_origin(), cache_key="updatedzd_z2"
    )
    copy_stencil(
        zh,
        z2,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )

    fvtp2d = utils.cached_stencil_class(FiniteVolumeTransport)(
        spec.namelist, spec.namelist.hord_tm, cache_key="updatedzd"
    )
    fvtp2d(
        z2,
        crx_adv,
        cry_adv,
        xfx_adv,
        yfx_adv,
        ra_x,
        ra_y,
        fx,
        fy,
    )
    # TODO, do not recreate this, and have it part of aninitialization step
    # or remove entirely when refactored away
    column_namelist = d_sw.get_column_namelist()
    for kstart, nk in d_sw.k_bounds():
        if column_namelist["damp_vt"][kstart] <= 1e-5:
            raise Exception("damp <= 1e-5 in column_cols is untested")
        delnflux.compute_no_sg(
            z2,
            fx2,
            fy2,
            int(column_namelist["nord_v"][kstart]),
            column_namelist["damp_vt"][kstart],
            wk,
            kstart=kstart,
            nk=nk,
        )
    zh_damp(
        grid.area,
        z2,
        fx,
        fy,
        ra_x,
        ra_y,
        fx2,
        fy2,
        grid.rarea,
        zh,
        zs,
        wsd,
        dt,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(0, 0, 1)),
    )
