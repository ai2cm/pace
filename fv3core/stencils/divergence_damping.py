import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, get_stencils_with_varied_bounds
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from fv3core.utils.grid import GridIndexing, axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
from fv3gfs.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    mintmp = min(0.2, dddmp * abs(q))
    damp = da_min_c * max(d2_bg, mintmp)
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
        self,
        grid_indexing: GridIndexing,
        agrid1,
        agrid2,
        bgrid1,
        bgrid2,
        dxa,
        dya,
        edge_n,
        edge_s,
        edge_e,
        edge_w,
        nested: bool,
        stretched_grid: bool,
        da_min,
        da_min_c,
        divg_u,
        divg_v,
        rarea_c,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cosa_u,
        cosa_v,
        sina_u,
        sina_v,
        dxc,
        dyc,
        dddmp,
        d4_bg,
        nord,
        grid_type,
        nord_col: FloatFieldK,
        d2_bg: FloatFieldK,
    ):
        self._idx = grid_indexing
        assert not nested, "nested not implemented"
        assert grid_type < 3, "Not implemented, grid_type>=3, specifically smag_corner"
        # TODO: make dddmp a compile-time external, instead of runtime scalar
        self._dddmp = dddmp
        # TODO: make da_min_c a compile-time external, instead of runtime scalar
        self._da_min_c = da_min_c
        self._grid_type = grid_type
        self._nord_column = nord_col
        self._d2_bg_column = d2_bg
        self._divg_u = divg_u
        self._divg_v = divg_v
        self._rarea_c = rarea_c
        self._sin_sg1 = sin_sg1
        self._sin_sg2 = sin_sg2
        self._sin_sg3 = sin_sg3
        self._sin_sg4 = sin_sg4
        self._cosa_u = cosa_u
        self._cosa_v = cosa_v
        self._sina_u = sina_u
        self._sina_v = sina_v
        self._dxc = dxc
        self._dyc = dyc
        nonzero_nord_k = 0
        self._nonzero_nord = int(nord)
        for k in range(len(self._nord_column)):
            if self._nord_column[k] > 0:
                nonzero_nord_k = k
                self._nonzero_nord = int(self._nord_column[k])
                break
        if stretched_grid:
            self._dd8 = da_min * d4_bg ** (self._nonzero_nord + 1)
        else:
            self._dd8 = (da_min_c * d4_bg) ** (self._nonzero_nord + 1)
        # TODO: make stretched_grid a compile-time external, instead of runtime scalar
        self._stretched_grid = stretched_grid
        kstart = nonzero_nord_k
        nk = self._idx.domain[2] - kstart
        low_k_idx = self._idx.restrict_vertical(k_start=0, nk=nonzero_nord_k)
        high_k_idx = grid_indexing.restrict_vertical(k_start=nonzero_nord_k)
        self.a2b_ord4 = AGrid2BGridFourthOrder(
            high_k_idx,
            agrid1,
            agrid2,
            bgrid1,
            bgrid2,
            dxa,
            dya,
            edge_n,
            edge_s,
            edge_e,
            edge_w,
            self._grid_type,
            replace=False,
        )

        origin, domain = low_k_idx.get_origin_domain(
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM], halos=(1, 0)
        )
        self._ptc_computation = FrozenStencil(
            ptc_computation,
            origin=origin,
            domain=domain,
            externals=axis_offsets(low_k_idx, origin, domain),
        )

        origin, domain = low_k_idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM], halos=(0, 1)
        )
        self._vorticity_computation = FrozenStencil(
            vorticity_computation,
            origin=origin,
            domain=domain,
            externals=axis_offsets(low_k_idx, origin, domain),
        )

        origin, domain = low_k_idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM], halos=(0, 0)
        )
        self._delpc_computation_and_damping = FrozenStencil(
            delpc_computation,
            origin=origin,
            domain=domain,
            externals=axis_offsets(self._idx, origin, domain),
        )
        self._damping = FrozenStencil(
            damping,
            origin=origin,
            domain=domain,
        )

        origin, domain = high_k_idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM], halos=(0, 0)
        )
        self._copy_computeplus = FrozenStencil(
            basic.copy_defn, origin=origin, domain=domain
        )

        origins = []
        origins_v = []
        origins_u = []
        domains = []
        domains_v = []
        domains_u = []
        for n in range(1, self._nonzero_nord + 1):
            nt = self._nonzero_nord - n
            nint = self._idx.domain[0] + 2 * nt + 1
            njnt = self._idx.domain[1] + 2 * nt + 1
            js = self._idx.jsc - nt
            is_ = self._idx.isc - nt
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

        origin, domain = high_k_idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM], halos=(0, 0)
        )
        self._damping_nord_highorder_stencil = FrozenStencil(
            damping_nord_highorder_stencil,
            origin=origin,
            domain=domain,
        )

        self._smagorinksy_diffusion_approx_stencil = FrozenStencil(
            smagorinksy_diffusion_approx,
            origin=origin,
            domain=domain,
        )

        origin, domain = high_k_idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            halos=(self._idx.n_halo, self._idx.n_halo),
        )
        self._set_value = FrozenStencil(
            basic.set_value_defn,
            origin=origin,
            domain=domain,
        )

        self._corner_tmp = utils.make_storage_from_shape(self._idx.max_shape)

        fill_origin, fill_domain = high_k_idx.get_origin_domain(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            halos=(self._idx.n_halo, self._idx.n_halo),
        )
        self.fill_corners_bgrid_x = corners.FillCornersBGrid(
            "x", self._corner_tmp, origin=fill_origin, domain=fill_domain
        )
        self.fill_corners_bgrid_y = corners.FillCornersBGrid(
            "y", self._corner_tmp, origin=fill_origin, domain=fill_domain
        )
        ax_offsets = axis_offsets(self._idx, fill_origin, fill_domain)
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
                    self._idx.sw_corner
                    or self._idx.se_corner
                    or self._idx.ne_corner
                    or self._idx.nw_corner
                )
            )
            if fillc:
                self.fill_corners_bgrid_x(
                    divg_d,
                )
            self._vc_from_divg_stencils[n](
                divg_d,
                self._divg_u,
                vc,
            )
            if fillc:
                self.fill_corners_bgrid_y(
                    divg_d,
                )
            self._uc_from_divg_stencils[n](
                divg_d,
                self._divg_v,
                uc,
            )

            if fillc:
                self._fill_corners_dgrid_stencil(
                    vc,
                    uc,
                    -1.0,
                )
            self._redo_divg_d_stencils[n](
                uc, vc, divg_d, self._rarea_c, self._stretched_grid
            )

        self.vorticity_calc(wk, vort, delpc, dt)
        self._damping_nord_highorder_stencil(
            vort,
            ke,
            delpc,
            divg_d,
            self._d2_bg_column,
            self._da_min_c,
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
            self._cosa_v,
            self._sina_v,
            self._dyc,
            self._sin_sg2,
            self._sin_sg4,
            ptc,
        )

        self._vorticity_computation(
            v,
            ua,
            self._cosa_u,
            self._sina_u,
            self._dxc,
            vort,
            uc,
            self._sin_sg3,
            self._sin_sg1,
        )
        # end if nested

        self._delpc_computation_and_damping(
            ptc,
            self._rarea_c,
            delpc,
            vort,
        )
        self._damping(
            delpc,
            vort,
            ke,
            d2_bg,
            self._da_min_c,
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
