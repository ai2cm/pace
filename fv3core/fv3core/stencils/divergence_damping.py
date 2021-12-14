import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
import pace.dsl.gt4py_utils as utils
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from fv3core.stencils.d2a2c_vect import contravariant
from fv3core.utils.grid import DampingCoefficients, GridData
from pace.dsl.stencil import StencilFactory, get_stencils_with_varied_bounds
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.util import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    mintmp = min(0.2, dddmp * abs(q))
    damp = da_min_c * max(d2_bg, mintmp)
    return damp


def get_delpc(
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    cosa_u: FloatFieldIJ,
    sina_u: FloatFieldIJ,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    sina_v: FloatFieldIJ,
    rarea_c: FloatFieldIJ,
    delpc: FloatField,
    u_contra_dyc: FloatField,
    v_contra_dxc: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start

    # in the Fortran, u_contra_dyc is called ke and v_contra_dxc is called vort

    with computation(PARALLEL), interval(...):
        # TODO: why does vc_from_va sometimes have different sign than vc?
        vc_from_va = 0.5 * (va[0, -1, 0] + va)
        # TODO: why do we use vc_from_va and not just vc?
        u_contra = contravariant(u, vc_from_va, cosa_v, sina_v)
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            u_contra = u * sin_sg4[0, -1] if vc > 0 else u * sin_sg2
        u_contra_dyc = u_contra * dyc

        # TODO: why does uc_from_ua sometimes have different sign than uc?
        uc_from_ua = 0.5 * (ua[-1, 0, 0] + ua)
        # TODO: why do we use uc_from_ua and not just uc?
        v_contra = contravariant(v, uc_from_ua, cosa_u, sina_u)
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            v_contra = v * sin_sg3[-1, 0] if uc > 0 else v * sin_sg1
        v_contra_dxc = v_contra * dxc
        with horizontal(
            region[i_start, j_end + 1],
            region[i_end + 1, j_end + 1],
            region[i_start, j_start - 1],
            region[i_end + 1, j_start - 1],
        ):
            # TODO: seems odd that this adjustment is only needed for `v_contra_dxc`
            # but is not needed for `u_contra_dyc`. Is this a bug? Add a comment
            # describing what this adjustment is doing and why.
            v_contra_dxc = 0.0

    with computation(PARALLEL), interval(...):
        delpc = (
            v_contra_dxc[0, -1, 0]
            - v_contra_dxc
            + u_contra_dyc[-1, 0, 0]
            - u_contra_dyc
        )
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
):
    from __externals__ import do_adjustment, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        divg_d = uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d = vc[-1, 0, 0] - vc - uc
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            divg_d = uc[0, -1, 0] + vc[-1, 0, 0] - vc
        if __INLINED(do_adjustment):
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
        stencil_factory: StencilFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        nested: bool,
        stretched_grid: bool,
        dddmp,
        d4_bg,
        nord,
        grid_type,
        nord_col: FloatFieldK,
        d2_bg: FloatFieldK,
    ):
        self.grid_indexing = stencil_factory.grid_indexing
        assert not nested, "nested not implemented"
        assert grid_type < 3, "Not implemented, grid_type>=3, specifically smag_corner"
        # TODO: make dddmp a compile-time external, instead of runtime scalar
        self._dddmp = dddmp
        # TODO: make da_min_c a compile-time external, instead of runtime scalar
        self._da_min_c = damping_coefficients.da_min_c
        self._grid_type = grid_type
        self._nord_column = nord_col
        self._d2_bg_column = d2_bg
        self._rarea_c = grid_data.rarea_c
        self._sin_sg1 = grid_data.sin_sg1
        self._sin_sg2 = grid_data.sin_sg2
        self._sin_sg3 = grid_data.sin_sg3
        self._sin_sg4 = grid_data.sin_sg4
        self._cosa_u = grid_data.cosa_u
        self._cosa_v = grid_data.cosa_v
        self._sina_u = grid_data.sina_u
        self._sina_v = grid_data.sina_v
        self._dxc = grid_data.dxc
        self._dyc = grid_data.dyc
        # TODO: maybe compute locally divg_* grid variables
        # They parallel the del6_* variables a lot
        # so may want to pair together if you move
        self._divg_u = damping_coefficients.divg_u
        self._divg_v = damping_coefficients.divg_v

        nonzero_nord_k = 0
        self._nonzero_nord = int(nord)
        for k in range(len(self._nord_column)):
            if self._nord_column[k] > 0:
                nonzero_nord_k = k
                self._nonzero_nord = int(self._nord_column[k])
                break
        if stretched_grid:
            self._dd8 = damping_coefficients.da_min * d4_bg ** (self._nonzero_nord + 1)
        else:
            self._dd8 = (damping_coefficients.da_min_c * d4_bg) ** (
                self._nonzero_nord + 1
            )
        kstart = nonzero_nord_k
        nk = self.grid_indexing.domain[2] - kstart
        self._do_zero_order = nonzero_nord_k > 0
        low_k_stencil_factory = stencil_factory.restrict_vertical(
            k_start=0, nk=nonzero_nord_k
        )
        high_k_stencil_factory = stencil_factory.restrict_vertical(
            k_start=nonzero_nord_k
        )
        self.a2b_ord4 = AGrid2BGridFourthOrder(
            stencil_factory=high_k_stencil_factory,
            grid_data=grid_data,
            grid_type=self._grid_type,
            replace=False,
        )

        self._get_delpc = low_k_stencil_factory.from_dims_halo(
            func=get_delpc,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
            skip_passes=("GreedyMerging",),
        )

        self._damping = low_k_stencil_factory.from_dims_halo(
            damping,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
        )

        self._copy_computeplus = high_k_stencil_factory.from_dims_halo(
            func=basic.copy_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
        )

        origins = []
        origins_v = []
        origins_u = []
        domains = []
        domains_v = []
        domains_u = []
        for n in range(1, self._nonzero_nord + 1):
            nt = self._nonzero_nord - n
            nint = self.grid_indexing.domain[0] + 2 * nt + 1
            njnt = self.grid_indexing.domain[1] + 2 * nt + 1
            js = self.grid_indexing.jsc - nt
            is_ = self.grid_indexing.isc - nt
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
            stencil_factory=stencil_factory,
        )

        self._uc_from_divg_stencils = get_stencils_with_varied_bounds(
            uc_from_divg,
            origins=origins_u,
            domains=domains_u,
            stencil_factory=stencil_factory,
        )

        self._redo_divg_d_stencils = get_stencils_with_varied_bounds(
            redo_divg_d,
            origins=origins,
            domains=domains,
            stencil_factory=stencil_factory,
            externals={"do_adjustment": not stretched_grid},
        )

        self._damping_nord_highorder_stencil = high_k_stencil_factory.from_dims_halo(
            func=damping_nord_highorder_stencil,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
        )
        self._smagorinksy_diffusion_approx_stencil = (
            high_k_stencil_factory.from_dims_halo(
                func=smagorinksy_diffusion_approx,
                compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
                compute_halos=(0, 0),
            )
        )

        self._set_value = high_k_stencil_factory.from_dims_halo(
            func=basic.set_value_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(self.grid_indexing.n_halo, self.grid_indexing.n_halo),
        )

        self._corner_tmp = utils.make_storage_from_shape(
            self.grid_indexing.max_shape, backend=stencil_factory.backend
        )

        self.fill_corners_bgrid_x = corners.FillCornersBGrid(
            direction="x",
            temporary_field=self._corner_tmp,
            stencil_factory=high_k_stencil_factory,
        )
        self.fill_corners_bgrid_y = corners.FillCornersBGrid(
            direction="y",
            temporary_field=self._corner_tmp,
            stencil_factory=high_k_stencil_factory,
        )
        self._fill_corners_dgrid_stencil = high_k_stencil_factory.from_dims_halo(
            func=corners.fill_corners_dgrid_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(self.grid_indexing.n_halo, self.grid_indexing.n_halo),
        )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        u_contra_dyc: FloatField,
        v_contra_dxc: FloatField,
        ua: FloatField,
        divg_d: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        wk: FloatField,
        dt: float,
    ) -> None:
        """
        Args:
            u (in)
            v (in)
            va (in)
            u_contra_dyc (out)
            v_contra_dxc (out)
            ua (in)
            divg_d (inout)
            vc (inout)
            uc (inout)
            delpc (out)
            ke: gets vort added to it (inout)
            wk: gets converted by a2b_ord4 and put into wk at end (in)
            dt: timestep (in)
        """
        # in the original Fortran, u_contra_dyc is "ptc" and v_contra_dxc is "vort"
        if self._do_zero_order:
            # TODO: delpc is an output of this but is never used. Inside the helper
            # function, use a stencil temporary or temporary storage instead
            self._damping_zero_order(
                u,
                v,
                va,
                u_contra_dyc,
                v_contra_dxc,
                ua,
                vc,
                uc,
                delpc,
                ke,
                self._d2_bg_column,
                dt,
            )
        self._copy_computeplus(divg_d, delpc)
        for n in range(self._nonzero_nord):
            fillc = (
                (n + 1 != self._nonzero_nord)
                and self._grid_type < 3
                and (
                    self.grid_indexing.sw_corner
                    or self.grid_indexing.se_corner
                    or self.grid_indexing.ne_corner
                    or self.grid_indexing.nw_corner
                )
            )
            if fillc:
                self.fill_corners_bgrid_x(divg_d)
            self._vc_from_divg_stencils[n](divg_d, self._divg_u, vc)
            if fillc:
                self.fill_corners_bgrid_y(divg_d)
            self._uc_from_divg_stencils[n](divg_d, self._divg_v, uc)

            # TODO(eddied): We pass the same fields 2x to avoid GTC validation errors
            if fillc:
                self._fill_corners_dgrid_stencil(vc, vc, uc, uc, -1.0)
            self._redo_divg_d_stencils[n](uc, vc, divg_d, self._rarea_c)

        self._vorticity_calc(wk, v_contra_dxc, delpc, dt)
        self._damping_nord_highorder_stencil(
            v_contra_dxc,
            ke,
            delpc,
            divg_d,
            self._d2_bg_column,
            self._da_min_c,
            self._dddmp,
            self._dd8,
        )

    def _damping_zero_order(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        u_contra_dyc: FloatField,
        v_contra_dxc: FloatField,
        ua: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        d2_bg: FloatFieldK,
        dt: float,
    ) -> None:
        """
        Args:
            u (in)
            v (in)
            va (in)
            ptc (out)
            vort (out)
            ua (in)
            vc (in)
            uc (in)
            delpc (out)
            ke: gets vort added to it (inout)
            d2_bg (in)
            dt: timestep in seconds
        """
        # TODO: convert ptc and vort to gt4py temporaries using selective validation
        # their outputs from get_delpc do not get used

        self._get_delpc(
            u,
            v,
            ua,
            va,
            self._cosa_u,
            self._sina_u,
            self._dxc,
            self._dyc,
            uc,
            vc,
            self._sin_sg1,
            self._sin_sg2,
            self._sin_sg3,
            self._sin_sg4,
            self._cosa_v,
            self._sina_v,
            self._rarea_c,
            delpc,
            u_contra_dyc,
            v_contra_dxc,
        )

        self._damping(
            delpc,
            v_contra_dxc,
            ke,
            d2_bg,
            self._da_min_c,
            self._dddmp,
            dt,
        )

    def _vorticity_calc(self, wk, vort, delpc, dt):
        if self._dddmp < 1e-5:
            self._set_value(vort, 0.0)
        else:
            self.a2b_ord4(wk, vort)
            self._smagorinksy_diffusion_approx_stencil(
                delpc,
                vort,
                abs(dt),
            )
