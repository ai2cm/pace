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
import fv3core.utils
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils import basic_operations
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


def ra_update(
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
def edge_profile(
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
    """
    Args:
        q1x: ???
        q2x: ???
        qe1x: ???
        qe2x: ???
        q1y: ???
        q2y: ???
        qe1y: ???
        qe2y: ???
        dp0 (in): Reference pressure for layer interfaces, assuming a globally uniform
            reference surface pressure. Used as an approximation of pressure,
            for efficiency.
    """
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


class UpdateDeltaZOnDGrid:
    """
    Fortran name is updatedzd.
    """

    def __init__(self, grid, column_namelist, k_bounds):
        self.grid = spec.grid
        self._column_namelist = column_namelist
        if any(
            column_namelist["damp_vt"][kstart] <= 1e-5
            for kstart in range(len(k_bounds))
        ):
            raise NotImplementedError("damp <= 1e-5 in column_cols is untested")
        self._k_bounds = k_bounds  # d_sw.k_bounds()
        largest_possible_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._crx_adv = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(0, -self.grid.halo, 0))
        )
        self._cry_adv = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(-self.grid.halo, 0, 0))
        )
        self._xfx_adv = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(0, -self.grid.halo, 0))
        )
        self._yfx_adv = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(-self.grid.halo, 0, 0))
        )
        self._ra_x = utils.make_storage_from_shape(
            largest_possible_shape,
            grid.compute_origin(add=(0, -self.grid.halo, 0)),
        )
        self._ra_y = utils.make_storage_from_shape(
            largest_possible_shape,
            grid.compute_origin(add=(-self.grid.halo, 0, 0)),
            cache_key="updatedzd_ra_y",
        )
        self._wk = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fx2 = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fy2 = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fx = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fy = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._z2 = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )

        self.finite_volume_transport = FiniteVolumeTransport(
            spec.namelist, spec.namelist.hord_tm
        )
        ax_offsets = fv3core.utils.axis_offsets(
            self.grid, self.grid.full_origin(), self.grid.domain_shape_full()
        )
        self._ra_update = FrozenStencil(
            ra_update,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
            externals=ax_offsets,
        )
        self._edge_profile = FrozenStencil(
            edge_profile,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
            externals=ax_offsets,
        )
        self._zh_damp = FrozenStencil(
            zh_damp,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 0, 1)),
        )

    def __call__(
        self,
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
        """
        Args:
            dp0: ???
            zs: ???
            zh: ???
            crx: Courant number in x-direction (??? what units)
            cry: Courant number in y-direction (??? what units)
            xfx: ???
            yfx: ???
            wsd: ???
            dt: ???
        """
        self._edge_profile(
            crx,
            xfx,
            self._crx_adv,
            self._xfx_adv,
            cry,
            yfx,
            self._cry_adv,
            self._yfx_adv,
            dp0,
        )
        self._ra_update(
            self.grid.area,
            self._xfx_adv,
            self._ra_x,
            self._yfx_adv,
            self._ra_y,
        )
        basic_operations.copy_stencil(
            zh,
            self._z2,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self.finite_volume_transport(
            self._z2,
            self._crx_adv,
            self._cry_adv,
            self._xfx_adv,
            self._yfx_adv,
            self._ra_x,
            self._ra_y,
            self._fx,
            self._fy,
        )
        for kstart, nk in self._k_bounds:
            delnflux.compute_no_sg(
                self._z2,
                self._fx2,
                self._fy2,
                int(self._column_namelist["nord_v"][kstart]),
                self._column_namelist["damp_vt"][kstart],
                self._wk,
                kstart=kstart,
                nk=nk,
            )
        self._zh_damp(
            self.grid.area,
            self._z2,
            self._fx,
            self._fy,
            self._ra_x,
            self._ra_y,
            self._fx2,
            self._fy2,
            self.grid.rarea,
            zh,
            zs,
            wsd,
            dt,
        )


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
    updatedzd = utils.cached_stencil_class(UpdateDeltaZOnDGrid)(
        spec.grid, d_sw.get_column_namelist(), d_sw.k_bounds()
    )
    updatedzd(dp0, zs, zh, crx, cry, xfx, yfx, wsd, dt)
