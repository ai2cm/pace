from types import SimpleNamespace

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval, horizontal, region

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, get_stencils_with_varied_bounds
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    tmpddd = dddmp * abs(q)
    mintmp = 0.2 if 0.2 < tmpddd else tmpddd
    maxd2 = d2_bg if d2_bg > mintmp else mintmp
    damp = da_min_c * maxd2
    return damp


def ptc_computation(
    u: FloatField,
    va: FloatField,
    vc: FloatField,
    cosa_v: FloatFieldIJ,
    sina_v: FloatFieldIJ,
    dyc: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    ptc: FloatField,
):
    """computation of pct"""
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        ptc = (u - 0.5 * (va[0, -1, 0] + va) * cosa_v) * dyc * sina_v
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            ptc = u * dyc * sin_sg4[0, -1] if vc > 0 else u * dyc * sin_sg2


def vorticity_computation(
    v: FloatField,
    ua: FloatField,
    cosa_u: FloatFieldIJ,
    sina_u: FloatFieldIJ,
    dxc: FloatFieldIJ,
    vort: FloatField,
    uc: FloatField,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
):
    """computation of the vorticity"""
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        vort = (v - 0.5 * (ua[-1, 0, 0] + ua) * cosa_u) * dxc * sina_u
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vort = v * dxc * sin_sg3[-1, 0] if uc > 0 else v * dxc * sin_sg1


def delpc_computation(
    ptc: FloatField,
    rarea_c: FloatFieldIJ,
    delpc: FloatField,
    vort: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        delpc = vort[0, -1, 0] - vort + ptc[-1, 0, 0] - ptc
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            delpc = ptc[-1, 0, 0] - ptc - vort
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            delpc = vort[0, -1, 0] + ptc[-1, 0, 0] - ptc
        delpc = rarea_c * delpc


def damping(
    delpc: FloatField,
    vort: FloatField,
    ke: FloatField,
    d2_bg: FloatFieldK,
    da_min_c: float,
    dddmp: float,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        delpcdt = delpc * dt
        damp = damp_tmp(delpcdt, da_min_c, d2_bg, dddmp)
        vort = damp * delpc
        ke += vort


def damping_nord_highorder_stencil(
    vort: FloatField,
    ke: FloatField,
    delpc: FloatField,
    divg_d: FloatField,
    d2_bg: FloatFieldK,
    da_min_c: float,
    dddmp: float,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        damp = damp_tmp(vort, da_min_c, d2_bg, dddmp)
        vort = damp * delpc + dd8 * divg_d
        ke = ke + vort


def vc_from_divg(divg_d: FloatField, divg_u: FloatFieldIJ, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc = (divg_d[1, 0, 0] - divg_d) * divg_u


def uc_from_divg(divg_d: FloatField, divg_v: FloatFieldIJ, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = (divg_d[0, 1, 0] - divg_d) * divg_v


def redo_divg_d(
    uc: FloatField,
    vc: FloatField,
    divg_d: FloatField,
    adjustment_factor: FloatFieldIJ,
    skip_adjustment: bool,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        divg_d = uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d = vc[-1, 0, 0] - vc - uc
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            divg_d = uc[0, -1, 0] + vc[-1, 0, 0] - vc
        if not skip_adjustment:
            divg_d = divg_d * adjustment_factor


def smagorinksy_diffusion_approx(delpc: FloatField, vort: FloatField, absdt: float):
    with computation(PARALLEL), interval(...):
        vort = absdt * (delpc ** 2.0 + vort ** 2.0) ** 0.5


class DivergenceDamping:
    """
    A large section in Fortran's d_sw that applies divergence damping
    """

    def __init__(
        self, namelist: SimpleNamespace, nord_col: FloatFieldK, d2_bg: FloatFieldK
    ):
        self.grid = spec.grid
        assert not self.grid.nested, "nested not implemented"
        assert (
            namelist.grid_type < 3
        ), "Not implemented, grid_type>=3, specifically smag_corner"
        self._dddmp = namelist.dddmp
        self._d4_bg = namelist.d4_bg
        self._grid_type = namelist.grid_type
        self._nord_column = nord_col
        self._d2_bg_column = d2_bg
        self._nonzero_nord_k = 0
        self._nonzero_nord = int(namelist.nord)
        for k in range(len(self._nord_column)):
            if self._nord_column[k] > 0:
                self._nonzero_nord_k = k
                self._nonzero_nord = int(self._nord_column[k])
                break
        if self.grid.stretched_grid:
            self._dd8 = self.grid.da_min * self._d4_bg ** (self._nonzero_nord + 1)
        else:
            self._dd8 = (self.grid.da_min_c * self._d4_bg) ** (self._nonzero_nord + 1)
        kstart = self._nonzero_nord_k
        low_kstart = 0
        nk = self.grid.npz - kstart
        low_nk = self._nonzero_nord_k
        self.a2b_ord4 = AGrid2BGridFourthOrder(
            self._grid_type, kstart, nk, replace=False
        )

        start_points = axis_offsets(
            self.grid,
            (self.grid.is_ - 1, self.grid.js, low_kstart),
            (self.grid.nic + 2, self.grid.njc + 1, low_nk),
        )
        self._ptc_computation = FrozenStencil(
            ptc_computation,
            origin=(self.grid.is_ - 1, self.grid.js, low_kstart),
            domain=(self.grid.nic + 2, self.grid.njc + 1, low_nk),
            externals=start_points,
        )

        vorticity_origin = (self.grid.is_, self.grid.js - 1, low_kstart)
        vorticity_domain = (
            self.grid.ie + 1 - self.grid.is_ + 1,
            self.grid.njc + 2,
            low_nk,
        )
        start_points = axis_offsets(
            self.grid,
            vorticity_origin,
            vorticity_domain,
        )
        self._vorticity_computation = FrozenStencil(
            vorticity_computation,
            origin=vorticity_origin,
            domain=vorticity_domain,
            externals=start_points,
        )

        delpc_origin = (self.grid.is_, self.grid.js, low_kstart)
        delpc_domain = (self.grid.nic + 1, self.grid.njc + 1, low_nk)
        start_points = axis_offsets(
            self.grid,
            delpc_origin,
            delpc_domain,
        )
        self._delpc_computation_and_damping = FrozenStencil(
            delpc_computation,
            origin=delpc_origin,
            domain=delpc_domain,
            externals=start_points,
        )
        self._damping = FrozenStencil(
            damping,
            origin=delpc_origin,
            domain=delpc_domain,
        )

        self._copy_computeplus = FrozenStencil(
            basic.copy_defn,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )

        origins = []
        origins_v = []
        origins_u = []
        domains = []
        domains_v = []
        domains_u = []
        for n in range(1, self._nonzero_nord + 1):
            nt = self._nonzero_nord - n
            nint = self.grid.nic + 2 * nt + 1
            njnt = self.grid.njc + 2 * nt + 1
            js = self.grid.js - nt
            is_ = self.grid.is_ - nt
            origins_v.append((is_ - 1, js, kstart))
            domains_v.append((nint + 1, njnt, nk))
            origins_u.append((is_, js - 1, kstart))
            domains_u.append((nint, njnt + 1, nk))
            origins.append((is_, js, kstart))
            domains.append((nint, njnt, nk))
        self._vc_from_divg_stencils = get_stencils_with_varied_bounds(
            vc_from_divg,
            origins=origins_v,
            domains=domains_v,
        )

        self._uc_from_divg_stencils = get_stencils_with_varied_bounds(
            uc_from_divg,
            origins=origins_u,
            domains=domains_u,
        )

        self._redo_divg_d_stencils = get_stencils_with_varied_bounds(
            redo_divg_d, origins=origins, domains=domains
        )

        self._damping_nord_highorder_stencil = FrozenStencil(
            damping_nord_highorder_stencil,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )
        self._smagorinksy_diffusion_approx_stencil = FrozenStencil(
            smagorinksy_diffusion_approx,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )
        self._set_value = FrozenStencil(
            basic.set_value_defn,
            origin=(self.grid.isd, self.grid.jsd, kstart),
            domain=(self.grid.nid + 1, self.grid.njd + 1, nk),
        )
        self._corner_tmp = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)), origin=self.grid.full_origin()
        )
        fill_origin = (self.grid.isd, self.grid.jsd, kstart)
        fill_domain = (self.grid.nid + 1, self.grid.njd + 1, nk)
        self.fill_corners_bgrid_x = corners.FillCornersBGrid(
            "x", self._corner_tmp, origin=fill_origin, domain=fill_domain
        )
        self.fill_corners_bgrid_y = corners.FillCornersBGrid(
            "y", self._corner_tmp, origin=fill_origin, domain=fill_domain
        )
        ax_offsets = axis_offsets(self.grid, fill_origin, fill_domain)
        self._fill_corners_dgrid_stencil = FrozenStencil(
            corners.fill_corners_dgrid_defn,
            externals=ax_offsets,
            origin=fill_origin,
            domain=fill_domain,
        )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        ptc: FloatField,
        vort: FloatField,
        ua: FloatField,
        divg_d: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        wk: FloatField,
        dt: float,
    ) -> None:

        self.damping_zero_order(
            u, v, va, ptc, vort, ua, vc, uc, delpc, ke, self._d2_bg_column, dt
        )
        self._copy_computeplus(
            divg_d,
            delpc,
        )
        for n in range(self._nonzero_nord):
            fillc = (
                (n + 1 != self._nonzero_nord)
                and self._grid_type < 3
                and (
                    self.grid.sw_corner
                    or self.grid.se_corner
                    or self.grid.ne_corner
                    or self.grid.nw_corner
                )
            )
            if fillc:
                self.fill_corners_bgrid_x(
                    divg_d,
                )
            self._vc_from_divg_stencils[n](
                divg_d,
                self.grid.divg_u,
                vc,
            )
            if fillc:
                self.fill_corners_bgrid_y(
                    divg_d,
                )
            self._uc_from_divg_stencils[n](
                divg_d,
                self.grid.divg_v,
                uc,
            )
            if fillc:
                self._fill_corners_dgrid_stencil(
                    vc,
                    uc,
                    -1.0,
                )
            self._redo_divg_d_stencils[n](
                uc, vc, divg_d, self.grid.rarea_c, self.grid.stretched_grid
            )

        self.vorticity_calc(wk, vort, delpc, dt)
        self._damping_nord_highorder_stencil(
            vort,
            ke,
            delpc,
            divg_d,
            self._d2_bg_column,
            self.grid.da_min_c,
            self._dddmp,
            self._dd8,
        )

    def damping_zero_order(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        ptc: FloatField,
        vort: FloatField,
        ua: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        d2_bg: FloatFieldK,
        dt: float,
    ) -> None:
        # if nested
        # TODO: ptc and vort are equivalent, but x vs y, consolidate if possible.
        self._ptc_computation(
            u,
            va,
            vc,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.dyc,
            self.grid.sin_sg2,
            self.grid.sin_sg4,
            ptc,
        )

        self._vorticity_computation(
            v,
            ua,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.dxc,
            vort,
            uc,
            self.grid.sin_sg3,
            self.grid.sin_sg1,
        )
        # end if nested

        self._delpc_computation_and_damping(
            ptc,
            self.grid.rarea_c,
            delpc,
            vort,
        )
        self._damping(
            delpc,
            vort,
            ke,
            d2_bg,
            self.grid.da_min_c,
            self._dddmp,
            dt,
        )

    def vorticity_calc(self, wk, vort, delpc, dt):
        if self._dddmp < 1e-5:
            self._set_value(vort, 0.0)
        else:
            self.a2b_ord4(wk, vort)
            self._smagorinksy_diffusion_approx_stencil(
                delpc,
                vort,
                abs(dt),
            )
