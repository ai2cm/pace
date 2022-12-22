import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import pace.fv3core.stencils.basic_operations as basic
import pace.stencils.corners as corners
import pace.util
from pace.dsl.dace.orchestration import dace_inhibitor, orchestrate
from pace.dsl.stencil import StencilFactory, get_stencils_with_varied_bounds
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from pace.fv3core.stencils.d2a2c_vect import contravariant
from pace.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from pace.util.grid import DampingCoefficients, GridData


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    mintmp = min(0.2, dddmp * abs(q))
    damp = da_min_c * max(d2_bg, mintmp)
    return damp


def compute_u_contra_dyc(
    u: FloatField,
    va: FloatField,
    vc: FloatField,
    cosa_v: FloatFieldIJ,
    sina_v: FloatFieldIJ,
    dyc: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    u_contra_dyc: FloatField,
):
    """

    Args:
        u (in):
        va (in):
        vc (in):
        cosa_v (in):
        sina_v (in):
        dyc (in):
        sin_sg2 (in):
        sin_sg4 (in):
        u_contra_dyc (out): contravariant u-wind on d-grid
    """
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        # TODO: why does vc_from_va sometimes have different sign than vc?
        vc_from_va = 0.5 * (va[0, -1, 0] + va)
        # TODO: why do we use vc_from_va and not just vc?
        u_contra = contravariant(u, vc_from_va, cosa_v, sina_v)
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            u_contra = u * sin_sg4[0, -1] if vc > 0 else u * sin_sg2
        u_contra_dyc = u_contra * dyc


def compute_v_contra_dxc(
    v: FloatField,
    ua: FloatField,
    cosa_u: FloatFieldIJ,
    sina_u: FloatFieldIJ,
    dxc: FloatFieldIJ,
    v_contra_dxc: FloatField,
    uc: FloatField,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
):
    """
    Args:
        v (in):
        ua (in):
        cosa_u (in):
        sina_u (in):
        dxc (in):
        vort (out): contravariant v-wind on d-grid
        uc (in):
        sin_sg3 (in):
        sin_sg1 (in):
    """
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        # TODO: why does uc_from_ua sometimes have different sign than uc?
        uc_from_ua = 0.5 * (ua[-1, 0, 0] + ua)
        # TODO: why do we use uc_from_ua and not just uc?
        v_contra = contravariant(v, uc_from_ua, cosa_u, sina_u)
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            v_contra = v * sin_sg3[-1, 0] if uc > 0 else v * sin_sg1
        v_contra_dxc = v_contra * dxc


def delpc_computation(
    u_contra_dxc: FloatField,
    rarea_c: FloatFieldIJ,
    delpc: FloatField,  # TODO: rename to divergence_on_cell_corners
    v_contra_dyc: FloatField,
):
    """
    Args:
        u_contra_dyc (in): contravariant u-wind on d-grid * dxc
        rarea_c (in):
        delpc (out): convergence of wind on cell centers
        v_contra_dxc (in): contravariant v-wind on d-grid * dyc
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        delpc = (
            v_contra_dyc[0, -1, 0]
            - v_contra_dyc
            + u_contra_dxc[-1, 0, 0]
            - u_contra_dxc
        )

    # dual quadrilateral becomes dual triangle, at the corners, so there is
    # an extraneous term in the divergence calculation. This is always
    # done using the y-component, though it could be done with either
    # the y- or x-component (they should be identical).
    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            delpc = delpc - v_contra_dyc[0, -1, 0]
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            delpc = delpc + v_contra_dyc

    with computation(PARALLEL), interval(...):
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
    """
    Args:
        delpc (in): divergence at cell corner
        vort (out):
        ke (inout):
        d2_bg (in):
    """
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
    """
    Args:
        vort (inout): linear combination of second-order and higher-order
            divergence damping, on output is the damping term itself
        ke (inout): on input, is the kinetic energy, on output also includes
            the damping term vort
        delpc (in): divergence on cell corners
        divg_d (in): higher-order divergence on d-grid
        d2_bg (in): background second-order divergence damping coefficient
    """
    # TODO: propagate variable renaming into this routine
    with computation(PARALLEL), interval(...):
        damp = damp_tmp(vort, da_min_c, d2_bg, dddmp)
        vort = damp * delpc + dd8 * divg_d
        ke = ke + vort


def vc_from_divg(divg_d: FloatField, divg_u: FloatFieldIJ, vc: FloatField):
    """
    Args:
        divg_d (in): divergence on d-grid
        divg_u (in): metric term, divg_u = sina_v * dyc / dx
        uv (out): intermediate component of hyperdiffusion defined on
            same grid as c-grid y-wind
    """
    with computation(PARALLEL), interval(...):
        vc = (divg_d[1, 0, 0] - divg_d) * divg_u


def uc_from_divg(divg_d: FloatField, divg_v: FloatFieldIJ, uc: FloatField):
    """
    Args:
        divg_d (in): divergence on d-grid
        divg_v (in): metric term, divg_v = sina_u * dxc / dy
        uc (out): intermediate component of hyperdiffusion defined on
            same grid as c-grid x-wind
    """
    with computation(PARALLEL), interval(...):
        uc = (divg_d[0, 1, 0] - divg_d) * divg_v


def redo_divg_d(
    uc: FloatField,
    vc: FloatField,
    divg_d: FloatField,
    adjustment_factor: FloatFieldIJ,
):
    """
    Args:
        uc (in): intermediate component of hyperdiffusion defined on
            same grid as c-grid x-wind
        vc (in): intermediate component of hyperdiffusion defined on
            same grid as c-grid y-wind
        divg_d (out): updated divergence for hyperdiffusion on d-grid
        adjustment_factor (in):
    """
    from __externals__ import do_adjustment, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        divg_d = uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d = divg_d - uc[0, -1, 0]
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            divg_d = divg_d + uc

    with computation(PARALLEL), interval(...):
        if __INLINED(do_adjustment):
            divg_d = divg_d * adjustment_factor


def smagorinsky_diffusion_approx(delpc: FloatField, vort: FloatField, absdt: float):
    """
    Args:
        delpc (in): divergence on cell corners
        vort (inout): local eddy diffusivity
        absdt (in): abs(dt)
    """
    with computation(PARALLEL), interval(...):
        vort = absdt * (delpc ** 2.0 + vort ** 2.0) ** 0.5


class DivergenceDamping:
    """
    A large section in Fortran's d_sw that applies divergence damping
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        nested: bool,
        stretched_grid: bool,
        dddmp,
        d4_bg,
        nord: int,
        grid_type,
        nord_col: pace.util.Quantity,
        d2_bg: FloatFieldK,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        self.grid_indexing = stencil_factory.grid_indexing
        assert not nested, "nested not implemented"
        assert grid_type < 3, "Not implemented, grid_type>=3, specifically smag_corner"
        # TODO: make dddmp a compile-time external, instead of runtime scalar
        self._dddmp = dddmp
        # TODO: make da_min_c a compile-time external, instead of runtime scalar
        self._damping_coefficients = damping_coefficients
        self._stretched_grid = stretched_grid
        self._d4_bg = d4_bg
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
        # everything below the sponge layer (k=3 to npz) would use nord, everything
        # within the sponge layer uses the same higher nord value equal to the
        # first nonzero value in nord_column
        # k = 1, 2 nord = 0
        # k = 3 to npz nord = user speicfied nord

        # refer to https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/main/model/dyn_core.F90#L693  # noqa: E501
        # for comparison
        self._nonzero_nord = int(nord)
        for k in range(self._nord_column.extent[0]):
            if self._nord_column.view[k] > 0:
                nonzero_nord_k = k
                self._nonzero_nord = int(self._nord_column.view[k])
                break

        kstart = nonzero_nord_k
        nk = self.grid_indexing.domain[2] - kstart
        self._do_zero_order = nonzero_nord_k > 0
        low_k_stencil_factory = stencil_factory.restrict_vertical(
            k_start=0, nk=nonzero_nord_k
        )
        high_k_stencil_factory = stencil_factory.restrict_vertical(
            k_start=nonzero_nord_k
        )

        self._compute_u_contra_dyc = low_k_stencil_factory.from_dims_halo(
            compute_u_contra_dyc,
            compute_dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(1, 0),
        )

        self._compute_v_contra_dxc = low_k_stencil_factory.from_dims_halo(
            compute_v_contra_dxc,
            compute_dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            compute_halos=(0, 1),
        )

        self._delpc_computation = low_k_stencil_factory.from_dims_halo(
            delpc_computation,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
        )

        self.u_contra_dyc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m^2/s")
        self.v_contra_dxc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m^2/s")

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
        self.fill_corners_bgrid_x = corners.FillCornersBGrid(
            direction="x",
            stencil_factory=high_k_stencil_factory,
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

        self.fill_corners_bgrid_y = corners.FillCornersBGrid(
            direction="y",
            stencil_factory=high_k_stencil_factory,
        )

        self._uc_from_divg_stencils = get_stencils_with_varied_bounds(
            uc_from_divg,
            origins=origins_u,
            domains=domains_u,
            stencil_factory=stencil_factory,
        )

        self._fill_corners_dgrid_stencil = high_k_stencil_factory.from_dims_halo(
            func=corners.fill_corners_dgrid_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(self.grid_indexing.n_halo, self.grid_indexing.n_halo),
        )

        self._redo_divg_d_stencils = get_stencils_with_varied_bounds(
            redo_divg_d,
            origins=origins,
            domains=domains,
            stencil_factory=stencil_factory,
            externals={"do_adjustment": not stretched_grid},
        )

        self._set_value = high_k_stencil_factory.from_dims_halo(
            func=basic.set_value_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(self.grid_indexing.n_halo, self.grid_indexing.n_halo),
        )

        self.a2b_ord4 = AGrid2BGridFourthOrder(
            stencil_factory=high_k_stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            grid_type=self._grid_type,
            replace=False,
        )

        self._smagorinksy_diffusion_approx_stencil = (
            high_k_stencil_factory.from_dims_halo(
                func=smagorinsky_diffusion_approx,
                compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
                compute_halos=(0, 0),
            )
        )

        self._damping_nord_highorder_stencil = high_k_stencil_factory.from_dims_halo(
            func=damping_nord_highorder_stencil,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            compute_halos=(0, 0),
        )

    # We need to use a getter for da_min & da_min_c in order to go around a DaCe inline
    # behavior. As part of the automatic optimization process, DaCe tries to inline
    # as many scalars as possible.
    # The grid is _not_ passed as an input to the top level function we orchestrate,
    # so its scalar values will be inlined.

    # 'alas, our distributed compilation system works by compiling a 3,3 layout
    # top tile, then using those 9 caches on every layout upward.
    # This setup leads to the values of da_min/da_min_c from the 3,3 layout
    # to be inlined in the generated code. Those variables are used in runtime
    # calculation (kinetic energy, etc.) which obviously leads to misbehaving numerics
    # and errors when the 3,3 layout values are used on larger layouts

    # The solution we implement here is making use of the fact that callbacks
    # are never inlined in dace optimization. the current workaround uses the
    # following functions.

    # An alternative would be to pass the Grid or the DampingCoefficients to DaCe,
    # clearly flagging it has a dynamic piece of memory (which would
    # cancel any inlining) but the feature to do that (dace.struct)
    # is currently in disarray.
    # N.B.: another solution is to pass da_min and da_min_c as input, put it seems
    # odd and adds a lot of boilerplate throughout the model code.

    @dace_inhibitor
    def _get_da_min_c(self) -> float:
        return self._damping_coefficients.da_min_c

    @dace_inhibitor
    def _get_da_min(self) -> float:
        return self._damping_coefficients.da_min

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        damped_rel_vort_bgrid: FloatField,
        ua: FloatField,
        divg_d: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        rel_vort_agrid: FloatField,
        dt: float,
    ):
        """
        Adds another form of diffusive flux that acts on the divergence field.
        To apply diffusion you can take the gradient of the divergence field and add
        it into the components of the velocity equation. This gives
        second-order diffusion, which is quite diffusive. But if we apply this
        iteratively, we can get 4th, 6th, 8th, etc. order diffusion.

        Explained in detail in section 8.3 of the FV3 documentation.

        Applies both a background second-order diffusion (with strength controlled by
        d2_bg passed on init) and a higher-order hyperdiffusion.

        Args:
            u (in): x-velocity on d-grid
            v (in): y-velocity on d-grid
            va (in):
            damped_rel_vort_bgrid (out): damped relative vorticity on b-grid
            ua (in):
            divg_d (inout): finite volume divergence defined on cell corners,
                output value is not used later in D_SW
            vc (inout):
            uc (inout):
            delpc (out): finite volume divergence defined on cell corners
            ke (inout): dt times the kinetic energy defined on cell corners,
                at input time must be accurate for the input winds.
                Gets updated to remain accurate for the output winds,
                as described in section 8.3 of the FV3 documentation.
            rel_vort_agrid (in): a-grid relative vorticity computed before
                divergence damping
                gets converted by a2b_ord4 and put into v_contra_dxc
            dt (in): timestep
        """
        # TODO: is there anything we can do to APIs to make it clear that divg_d is not
        #       really an output variable of DivergenceDamping?
        # in the original Fortran, u_contra_dyc is "ptc" and v_contra_dxc is "vort"
        # TODO: what does do_zero_order signify, why is it false/true?
        if self._do_zero_order:
            # This is used in the sponge layer, 2nd order damping
            # TODO: delpc is an output of this but is never used. Inside the helper
            # function, use a stencil temporary or temporary storage instead
            self._compute_u_contra_dyc(
                u,
                va,
                vc,
                self._cosa_v,
                self._sina_v,
                self._dyc,
                self._sin_sg2,
                self._sin_sg4,
                self.u_contra_dyc,
            )

            self._compute_v_contra_dxc(
                v,
                ua,
                self._cosa_u,
                self._sina_u,
                self._dxc,
                self.v_contra_dxc,
                uc,
                self._sin_sg3,
                self._sin_sg1,
            )

            self._delpc_computation(
                self.u_contra_dyc,
                self._rarea_c,
                delpc,
                self.v_contra_dxc,
            )

            da_min_c: float = self._get_da_min_c()
            self._damping(
                delpc,
                damped_rel_vort_bgrid,
                ke,
                self._d2_bg_column,
                da_min_c,
                self._dddmp,
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

        if self._dddmp < 1e-5:
            self._set_value(damped_rel_vort_bgrid, 0.0)
        else:
            # TODO: what is wk/v_contra_dxc here?
            # take the cell centered relative vorticity and regrid it to cell corners
            # for smagorinsky diffusion
            #
            self.a2b_ord4(rel_vort_agrid, damped_rel_vort_bgrid)
            self._smagorinksy_diffusion_approx_stencil(
                delpc,
                damped_rel_vort_bgrid,
                abs(dt),
            )

        da_min: float = self._get_da_min()
        if self._stretched_grid:
            # reference https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/main/model/sw_core.F90#L1422 # noqa: E501
            dd8 = da_min * self._d4_bg ** (self._nonzero_nord + 1)
        else:
            dd8 = (da_min_c * self._d4_bg) ** (self._nonzero_nord + 1)

        self._damping_nord_highorder_stencil(
            damped_rel_vort_bgrid,
            ke,
            delpc,
            divg_d,
            self._d2_bg_column,
            da_min_c,
            self._dddmp,
            dd8,
        )
