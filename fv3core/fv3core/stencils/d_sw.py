import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core.stencils.delnflux as delnflux
import pace.dsl.gt4py_utils as utils
import pace.util.constants as constants
from fv3core._config import DGridShallowWaterLagrangianDynamicsConfig
from fv3core.stencils.basic_operations import compute_coriolis_parameter_defn
from fv3core.stencils.d2a2c_vect import contravariant
from fv3core.stencils.delnflux import DelnFluxNoSG
from fv3core.stencils.divergence_damping import DivergenceDamping
from fv3core.stencils.fvtp2d import (
    FiniteVolumeTransport,
    PreAllocatedCopiedCornersFactory,
)
from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.stencils.xtp_u import advect_u_along_x
from fv3core.stencils.ytp_v import advect_v_along_y
from fv3core.utils.grid import DampingCoefficients, GridData
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.util import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)


dcon_threshold = 1e-5


@gtscript.function
def flux_increment(gx, gy, rarea):
    """
    Args:
        gx: x-direction flux of some scalar q in units of q * area
            defined on cell interfaces
        gy: y-direction flux of some scalar q in units of q * area
            defined on cell interfaces
        rarea: 1 / area

    Returns:
        tendency increment in units of q defined on cell centers
    """
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


def flux_adjust(
    q: FloatField, delp: FloatField, gx: FloatField, gy: FloatField, rarea: FloatFieldIJ
):
    """
    Update q according to fluxes gx and gy.

    Args:
        q: any scalar, is replaced with something in units of q * delp (inout)
        delp: pressure thickness of layer (in)
        gx: x-flux of q in units of q * Pa * area (in)
        gy: y-flux of q in units of q * Pa * area (in)
        rarea: 1 / area (in)
    """
    # TODO: this function changes the units and therefore meaning of q,
    # is there any way we can avoid doing so?
    # the next time w and q_con (passed as q to this routine) are used
    # is in adjust_w_and_qcon, where they are divided by an updated delp to return
    # to the original units.
    with computation(PARALLEL), interval(...):
        # in the original Fortran, this uses `w` instead of `q`
        q = q * delp + flux_increment(gx, gy, rarea)


def flux_capacitor(
    cx: FloatField,
    cy: FloatField,
    xflux: FloatField,
    yflux: FloatField,
    crx_adv: FloatField,
    cry_adv: FloatField,
    fx: FloatField,
    fy: FloatField,
):
    """Accumulates the flux capacitor and courant number variables
    Saves the mass fluxes to the "flux capacitor" variables for tracer transport
    Also updates the accumulated courant numbers
    Args:
        cx: accumulated courant number in the x direction (inout)
        cy: accumulated courant number in the y direction (inout)
        xflux: flux capacitor in the x direction, accumlated mass flux (inout)
        yflux: flux capacitor in the y direction, accumlated mass flux (inout)
        crx_adv: local courant numver, dt*ut/dx  (in)
        cry_adv: local courant number dt*vt/dy (in)
        fx: 1-D x-direction flux (in)
        fy: 1-D y-direction flux (in)
    """
    with computation(PARALLEL), interval(...):
        cx = cx + crx_adv
        cy = cy + cry_adv
        xflux = xflux + fx
        yflux = yflux + fy


@gtscript.function
def corner_ke(
    u,
    v,
    ut,
    vt,
    dt,
    io1,
    jo1,
    io2,
    vsign,
):
    dt6 = dt / 6.0

    return dt6 * (
        (ut[0, 0, 0] + ut[0, -1, 0]) * ((io1 + 1) * u[0, 0, 0] - (io1 * u[-1, 0, 0]))
        + (vt[0, 0, 0] + vt[-1, 0, 0]) * ((jo1 + 1) * v[0, 0, 0] - (jo1 * v[0, -1, 0]))
        + (
            ((jo1 + 1) * ut[0, 0, 0] - (jo1 * ut[0, -1, 0]))
            + vsign * ((io1 + 1) * vt[0, 0, 0] - (io1 * vt[-1, 0, 0]))
        )
        * ((io2 + 1) * u[0, 0, 0] - (io2 * u[-1, 0, 0]))
    )


@gtscript.function
def all_corners_ke(ke, u, v, ut, vt, dt):
    from __externals__ import i_end, i_start, j_end, j_start

    # Assumption: not __INLINED(grid.nested)
    with horizontal(region[i_start, j_start]):
        ke = corner_ke(u, v, ut, vt, dt, 0, 0, -1, 1)
    with horizontal(region[i_end + 1, j_start]):
        ke = corner_ke(u, v, ut, vt, dt, -1, 0, 0, -1)
    with horizontal(region[i_end + 1, j_end + 1]):
        ke = corner_ke(u, v, ut, vt, dt, -1, -1, 0, 1)
    with horizontal(region[i_start, j_end + 1]):
        ke = corner_ke(u, v, ut, vt, dt, 0, -1, -1, -1)

    return ke


@gtscript.function
def apply_pt_delp_fluxes(
    pt_x_flux: FloatField,
    pt_y_flux: FloatField,
    rarea: FloatFieldIJ,
    delp_x_flux: FloatField,
    delp_y_flux: FloatField,
    pt: FloatField,
    delp: FloatField,
):
    from __externals__ import inline_q, local_ie, local_is, local_je, local_js

    # original Fortran uses gx/gy for pt fluxes, fx/fy for delp fluxes
    # TODO: local region only needed for d_sw halo validation
    # use selective validation instead
    if __INLINED(inline_q == 0):
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
            pt = pt * delp + flux_increment(pt_x_flux, pt_y_flux, rarea)
            delp = delp + flux_increment(delp_x_flux, delp_y_flux, rarea)
            pt = pt / delp
    return pt, delp


def apply_pt_delp_fluxes_stencil_defn(
    fx: FloatField,
    fy: FloatField,
    pt: FloatField,
    delp: FloatField,
    gx: FloatField,
    gy: FloatField,
    rarea: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        pt, delp = apply_pt_delp_fluxes(gx, gy, rarea, fx, fy, pt, delp)


def kinetic_energy_update_part_1(
    vc: FloatField,
    uc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    v: FloatField,
    vc_contra: FloatField,
    u: FloatField,
    uc_contra: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    rdy: FloatFieldIJ,
    ub_contra: FloatField,
    vb_contra: FloatField,
    advected_u: FloatField,
    advected_v: FloatField,
    dt: float,
):

    with computation(PARALLEL), interval(...):
        ub_contra, vb_contra = interpolate_uc_vc_to_cell_corners(
            uc, vc, cosa, rsina, uc_contra, vc_contra
        )
        advected_v = advect_v_along_y(v, vb_contra, rdy=rdy, dy=dy, dya=dya, dt=dt)
        advected_u = advect_u_along_x(u, ub_contra, rdx=rdx, dx=dx, dxa=dxa, dt=dt)


def kinetic_energy_update_part_2(
    ub_contra: FloatField,
    advected_u: FloatField,
    vb_contra: FloatField,
    advected_v: FloatField,
    v: FloatField,
    vc_contra: FloatField,
    u: FloatField,
    uc_contra: FloatField,
    ke: FloatField,
    dt: float,
):
    # TODO: this is split from part 1 only because the compiled code asks for
    # too much memory. Merge these when that's fixed.
    with computation(PARALLEL), interval(...):
        # TODO: we see here that ke is not kinetic energy, but kinetic energy * timestep
        #       refactor or rename to avoid this confusion
        ke = 0.5 * dt * (ub_contra * advected_u + vb_contra * advected_v)
        ke = all_corners_ke(ke, u, v, uc_contra, vc_contra, dt)


def vort_differencing(
    vort: FloatField,
    vort_x_delta: FloatField,
    vort_y_delta: FloatField,
    dcon: FloatFieldK,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        if dcon[0] > dcon_threshold:
            # Creating a gtscript function for the ub/vb computation
            # results in an "NotImplementedError" error for Jenkins
            # Inlining the ub/vb computation in this stencil resolves the Jenkins error
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                vort_x_delta = vort - vort[1, 0, 0]
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                vort_y_delta = vort - vort[0, 1, 0]


@gtscript.function
def u_from_ke(ke, u, dx, fy):
    return u * dx + ke - ke[1, 0, 0] + fy


@gtscript.function
def v_from_ke(ke, v, dy, fx):
    return v * dy + ke - ke[0, 1, 0] - fx


def u_and_v_from_ke(
    ke: FloatField,
    fx: FloatField,
    fy: FloatField,
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    # TODO: this function does not return u and v, it returns something
    # like u * dx and v * dy. Rename this function and its inouts.

    with computation(PARALLEL), interval(...):
        # TODO: may be able to remove local regions once this stencil and
        # heat_from_damping are in the same stencil
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            u = u_from_ke(ke, u, dx, fy)
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            v = v_from_ke(ke, v, dy, fx)


# TODO: This is untested and the radius may be incorrect
@gtscript.function
def coriolis_force_correction(zh, radius):
    return 1.0 + (zh + zh[0, 0, 1]) / radius


def compute_vort(
    wk: FloatField,
    f0: FloatFieldIJ,
    zh: FloatField,
    vort: FloatField,
):

    from __externals__ import do_f3d, hydrostatic, radius

    with computation(PARALLEL), interval(...):
        if __INLINED(do_f3d and not hydrostatic):
            z_rat = coriolis_force_correction(zh, radius)
            vort = wk + f0 * z_rat
        else:
            vort = wk[0, 0, 0] + f0[0, 0]


def adjust_w_and_qcon(
    w: FloatField,
    delp: FloatField,
    dw: FloatField,
    q_con: FloatField,
    damp_w: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = w + dw if damp_w > 1e-5 else w
        # Fortran: #ifdef USE_COND
        q_con = q_con / delp


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


def heat_diss(
    fx2: FloatField,
    fy2: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    heat_source: FloatField,
    diss_est: FloatField,
    dw: FloatField,
    damp_w: FloatFieldK,
    ke_bg: FloatFieldK,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        diss_e = diss_est
        if damp_w > 1e-5:
            dd8 = ke_bg * abs(dt)
            dw = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
            heat_source = dd8 - dw * (w + 0.5 * dw)
            diss_est = diss_e + heat_source


def heat_source_from_vorticity_damping(
    vort_x_delta: FloatField,
    vort_y_delta: FloatField,
    ut: FloatField,
    vt: FloatField,
    u: FloatField,
    v: FloatField,
    delp: FloatField,
    rsin2: FloatFieldIJ,
    cosa_s: FloatFieldIJ,
    rdx: FloatFieldIJ,
    rdy: FloatFieldIJ,
    heat_source: FloatField,
    heat_source_total: FloatField,
    dissipation_estimate: FloatField,
    kinetic_energy_fraction_to_damp: FloatFieldK,
):
    """
    Calculates heat source from vorticity damping implied by energy conservation.
    Args:
        vort_x_delta (in)
        vort_y_delta (in)
        ut (in)
        vt (in)
        u (in)
        v (in)
        delp (in)
        rsin2 (in)
        cosa_s (in)
        rdx (in): 1 / dx
        rdy (in): 1 / dy
        heat_source (out): heat source from vorticity damping
            implied by energy conservation
        heat_source_total: (out) accumulated heat source
        dissipation_estimate (out): dissipation estimate, only calculated if
            calculate_dissipation_estimate is 1
        kinetic_energy_fraction_to_damp (in): according to its comment in fv_arrays,
            the fraction of kinetic energy to explicitly damp and convert into heat.
            TODO: confirm this description is accurate, why is it multiplied
            by 0.25 below?
    """
    from __externals__ import d_con, do_skeb, local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        ubt = (vort_x_delta + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vort_y_delta - ut) * rdy
        fx = v * rdy
        gx = fx * vbt

        if (kinetic_energy_fraction_to_damp > dcon_threshold) or do_skeb:
            u2 = fy + fy[0, 1, 0]
            du2 = ubt + ubt[0, 1, 0]
            v2 = fx + fx[1, 0, 0]
            dv2 = vbt + vbt[1, 0, 0]
            dampterm = heat_damping_term(
                ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2
            )
            heat_source = delp * (
                heat_source - 0.25 * kinetic_energy_fraction_to_damp * dampterm
            )

        if __INLINED((d_con > dcon_threshold) or do_skeb):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
                heat_source_total = heat_source_total + heat_source
                # TODO: do_skeb could be renamed to calculate_dissipation_estimate
                if __INLINED(do_skeb):
                    dissipation_estimate -= dampterm


# TODO(eddied): Had to split this into a separate stencil to get this to validate
#               with GTC, suspect a merging issue...
def update_u_and_v(
    ut: FloatField,
    vt: FloatField,
    u: FloatField,
    v: FloatField,
    damp_vt: FloatFieldK,
):
    """
    Updates u and v after calculation of heat source from vorticity damping.
    Args:
        ut (in)
        vt (in)
        u (in/out)
        v (in/out)
        damp_vt (in): column scalar for damping vorticity
    """

    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        if damp_vt > 1e-5:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                u += vt
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                v -= ut


def compute_vorticity(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    rarea: FloatFieldIJ,
    vorticity: FloatField,
):

    with computation(PARALLEL), interval(...):
        # TODO: ask Lucas why vorticity is computed with this particular treatment
        # of dx, dy, and rarea. The original code read like:
        #     u_dx = u * dx
        #     v_dy = v * dy
        #     vorticity = rarea * (u_dx - u_dx[0, 1, 0] - v_dy + v_dy[1, 0, 0])
        rdy_tmp = rarea * dx
        rdx_tmp = rarea * dy
        vorticity = (u - u[0, 1, 0] * dx[0, 1] / dx) * rdy_tmp + (
            v[1, 0, 0] * dy[1, 0] / dy - v
        ) * rdx_tmp


# Set the unique parameters for the smallest
# k-values, e.g. k = 0, 1, 2 when generating
# the column namelist
def set_low_kvals(col, k):
    for name in ["nord", "nord_w", "d_con"]:
        col[name][k] = 0
    col["damp_w"][k] = col["d2_divg"][k]


# For the column namelist at a specific k-level
# set the vorticity parameters if do_vort_damp is true
def vorticity_damping_option(column, k, do_vort_damp):
    if do_vort_damp:
        column["nord_v"][k] = 0
        column["damp_vt"][k] = 0.5 * column["d2_divg"][k]


def lowest_kvals(column, k, do_vort_damp):
    set_low_kvals(column, k)
    vorticity_damping_option(column, k, do_vort_damp)


def get_column_namelist(
    config: DGridShallowWaterLagrangianDynamicsConfig, npz, backend: str
):
    """
    Generate a dictionary of columns that specify how parameters (such as nord, damp)
    used in several functions called by D_SW vary over the k-dimension.

    """
    direct_namelist = ["ke_bg", "d_con", "nord"]
    all_names = direct_namelist + [
        "nord_v",
        "nord_w",
        "nord_t",
        "damp_vt",
        "damp_w",
        "damp_t",
        "d2_divg",
    ]
    col = {}
    for name in all_names:
        col[name] = utils.make_storage_from_shape((npz + 1,), (0,), backend=backend)
    for name in direct_namelist:
        col[name][:] = getattr(config, name)

    col["d2_divg"][:] = min(0.2, config.d2_bg)
    col["nord_v"][:] = min(2, col["nord"][0])
    col["nord_w"][:] = col["nord_v"][0]
    col["nord_t"][:] = col["nord_v"][0]
    if config.do_vort_damp:
        col["damp_vt"][:] = config.vtdm4
    else:
        col["damp_vt"][:] = 0
    col["damp_w"][:] = col["damp_vt"][0]
    col["damp_t"][:] = col["damp_vt"][0]
    if npz == 1 or config.n_sponge < 0:
        col["d2_divg"][0] = config.d2_bg
    else:
        col["d2_divg"][0] = max(0.01, config.d2_bg, config.d2_bg_k1)
        lowest_kvals(col, 0, config.do_vort_damp)
        if config.d2_bg_k2 > 0.01:
            col["d2_divg"][1] = max(config.d2_bg, config.d2_bg_k2)
            lowest_kvals(col, 1, config.do_vort_damp)
        if config.d2_bg_k2 > 0.05:
            col["d2_divg"][2] = max(config.d2_bg, 0.2 * config.d2_bg_k2)
            set_low_kvals(col, 2)
    return col


@gtscript.function
def interpolate_uc_vc_to_cell_corners(
    uc_cov, vc_cov, cosa, rsina, uc_contra, vc_contra
):
    """
    Convert covariant C-grid winds to contravariant B-grid (cell-corner) winds.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # In the original Fortran, this routine was given dt4 (0.25 * dt)
    # and dt5 (0.5 * dt), and its outputs were wind times timestep. This has
    # been refactored so the timestep is later explicitly multiplied, when
    # the wind is integrated forward in time.
    # TODO: ask Lucas why we interpolate then convert to contravariant in tile center,
    # but convert to contravariant and then interpolate on tile edges.
    ub_cov = 0.5 * (uc_cov[0, -1, 0] + uc_cov)
    vb_cov = 0.5 * (vc_cov[-1, 0, 0] + vc_cov)
    ub_contra = contravariant(ub_cov, vb_cov, cosa, rsina)
    vb_contra = contravariant(vb_cov, ub_cov, cosa, rsina)
    # ASSUME : if __INLINED(namelist.grid_type < 3):
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        ub_contra = 0.25 * (
            -uc_contra[0, -2, 0]
            + 3.0 * (uc_contra[0, -1, 0] + uc_contra)
            - uc_contra[0, 1, 0]
        )
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        ub_contra = 0.5 * (uc_contra[0, -1, 0] + uc_contra)
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        vb_contra = 0.25 * (
            -vc_contra[-2, 0, 0]
            + 3.0 * (vc_contra[-1, 0, 0] + vc_contra)
            - vc_contra[1, 0, 0]
        )
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        vb_contra = 0.5 * (vc_contra[-1, 0, 0] + vc_contra)

    return ub_contra, vb_contra


def compute_f0(
    stencil_factory: StencilFactory, lon_agrid: FloatFieldIJ, lat_agrid: FloatFieldIJ
):
    """
    Compute the coriolis parameter on the D-grid
    """
    f0 = utils.make_storage_from_shape(lon_agrid.shape, backend=stencil_factory.backend)
    f0_stencil = stencil_factory.from_dims_halo(
        compute_coriolis_parameter_defn,
        compute_dims=[X_DIM, Y_DIM, Z_DIM],
        compute_halos=(3, 3),
    )
    f0_stencil(f0, lon_agrid, lat_agrid, 0.0)
    return f0


class DGridShallowWaterLagrangianDynamics:
    """
    Fortran name is the d_sw subroutine
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        column_namelist,
        nested: bool,
        stretched_grid: bool,
        config: DGridShallowWaterLagrangianDynamicsConfig,
    ):
        self.grid_data = grid_data
        self._f0 = compute_f0(
            stencil_factory, self.grid_data.lon_agrid, self.grid_data.lat_agrid
        )

        self.grid_indexing = stencil_factory.grid_indexing
        assert config.grid_type < 3, "ubke and vbke only implemented for grid_type < 3"
        assert not config.inline_q, "inline_q not yet implemented"
        assert (
            config.d_ext <= 0
        ), "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        assert (column_namelist["damp_vt"] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_vt > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true
        assert (column_namelist["damp_w"] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_w > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true

        # only compute for k-levels where this is true
        self.hydrostatic = config.hydrostatic

        def make_storage():
            return utils.make_storage_from_shape(
                self.grid_indexing.max_shape, backend=stencil_factory.backend
            )

        self._tmp_heat_s = make_storage()
        self._vort_x_delta = make_storage()
        self._vort_y_delta = make_storage()
        self._tmp_ke = make_storage()
        self._tmp_vort = make_storage()
        self._uc_contra = make_storage()
        self._vc_contra = make_storage()
        self._tmp_ut = make_storage()
        self._tmp_vt = make_storage()
        self._tmp_fx = make_storage()
        self._tmp_fy = make_storage()
        self._tmp_gx = make_storage()
        self._tmp_gy = make_storage()
        self._tmp_dw = make_storage()
        self._tmp_wk = make_storage()
        self._tmp_fx2 = make_storage()
        self._tmp_fy2 = make_storage()
        self._tmp_damp_3d = utils.make_storage_from_shape(
            (1, 1, self.grid_indexing.domain[2]), backend=stencil_factory.backend
        )
        self._advected_u = make_storage()
        self._advected_v = make_storage()
        self._ub_contra = make_storage()
        self._vb_contra = make_storage()
        self._column_namelist = column_namelist

        self.delnflux_nosg_w = DelnFluxNoSG(
            stencil_factory,
            damping_coefficients,
            grid_data.rarea,
            self._column_namelist["nord_w"],
        )
        self.delnflux_nosg_v = DelnFluxNoSG(
            stencil_factory,
            damping_coefficients,
            grid_data.rarea,
            self._column_namelist["nord_v"],
        )
        self.fvtp2d_dp = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_dp,
            nord=self._column_namelist["nord_v"],
            damp_c=self._column_namelist["damp_vt"],
        )
        self.fvtp2d_dp_t = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_dp,
            nord=self._column_namelist["nord_t"],
            damp_c=self._column_namelist["damp_t"],
        )
        self.fvtp2d_tm = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_tm,
            nord=self._column_namelist["nord_v"],
            damp_c=self._column_namelist["damp_vt"],
        )
        self.fvtp2d_vt_nodelnflux = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_vt,
        )
        self.fv_prep = FiniteVolumeFluxPrep(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
        )
        self.divergence_damping = DivergenceDamping(
            stencil_factory,
            grid_data,
            damping_coefficients,
            nested,
            stretched_grid,
            config.dddmp,
            config.d4_bg,
            config.nord,
            config.grid_type,
            column_namelist["nord"],
            column_namelist["d2_divg"],
        )

        self._apply_pt_delp_fluxes = stencil_factory.from_dims_halo(
            func=apply_pt_delp_fluxes_stencil_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "inline_q": config.inline_q,
            },
        )
        self._kinetic_energy_update_part_1 = stencil_factory.from_dims_halo(
            func=kinetic_energy_update_part_1,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "iord": config.hord_mt,
                "jord": config.hord_mt,
                "mord": config.hord_mt,
                "xt_minmax": False,
                "yt_minmax": False,
            },
            skip_passes=("GreedyMerging",),
        )
        self._kinetic_energy_update_part_2 = stencil_factory.from_dims_halo(
            func=kinetic_energy_update_part_2,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "iord": config.hord_mt,
                "jord": config.hord_mt,
                "mord": config.hord_mt,
                "xt_minmax": False,
                "yt_minmax": False,
            },
            skip_passes=("GreedyMerging",),
        )
        self._flux_adjust_stencil = stencil_factory.from_dims_halo(
            func=flux_adjust, compute_dims=[X_DIM, Y_DIM, Z_DIM]
        )
        self._flux_capacitor_stencil = stencil_factory.from_dims_halo(
            func=flux_capacitor,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )
        self._vort_differencing_stencil = stencil_factory.from_dims_halo(
            func=vort_differencing,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        )
        self._u_and_v_from_ke_stencil = stencil_factory.from_dims_halo(
            func=u_and_v_from_ke, compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM]
        )
        self._compute_vort_stencil = stencil_factory.from_dims_halo(
            func=compute_vort,
            externals={
                "radius": constants.RADIUS,
                "do_f3d": config.do_f3d,
                "hydrostatic": self.hydrostatic,
            },
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )
        self._adjust_w_and_qcon_stencil = stencil_factory.from_dims_halo(
            func=adjust_w_and_qcon,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._heat_diss_stencil = stencil_factory.from_dims_halo(
            func=heat_diss,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._heat_source_from_vorticity_damping_stencil = (
            stencil_factory.from_dims_halo(
                func=heat_source_from_vorticity_damping,
                compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
                externals={
                    "do_skeb": config.do_skeb,
                    "d_con": config.d_con,
                },
            )
        )
        self._compute_vorticity_stencil = stencil_factory.from_dims_halo(
            compute_vorticity,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )
        self._update_u_and_v_stencil = stencil_factory.from_dims_halo(
            update_u_and_v, compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM]
        )
        damping_factor_calculation_stencil = stencil_factory.from_origin_domain(
            delnflux.calc_damp,
            origin=(0, 0, 0),
            domain=(1, 1, stencil_factory.grid_indexing.domain[2]),
        )
        damping_factor_calculation_stencil(
            self._tmp_damp_3d,
            self._column_namelist["nord_v"],
            self._column_namelist["damp_vt"],
            damping_coefficients.da_min_c,
        )
        self._delnflux_damp_vt = utils.make_storage_data(
            self._tmp_damp_3d[0, 0, :],
            (self.grid_indexing.domain[2],),
            (0,),
            backend=stencil_factory.backend,
        )

        damping_factor_calculation_stencil(
            self._tmp_damp_3d,
            self._column_namelist["nord_w"],
            self._column_namelist["damp_w"],
            damping_coefficients.da_min_c,
        )
        self._delnflux_damp_w = utils.make_storage_data(
            self._tmp_damp_3d[0, 0, :],
            (self.grid_indexing.domain[2],),
            (0,),
            backend=stencil_factory.backend,
        )
        y_temporary = utils.make_storage_from_shape(
            shape=self.grid_indexing.max_shape,
            origin=self.grid_indexing.origin,
            backend=stencil_factory.backend,
        )
        self._copy_corners = PreAllocatedCopiedCornersFactory(
            stencil_factory,
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
            y_temporary=y_temporary,
        )

    def __call__(
        self,
        delpc,
        delp,
        ptc,
        pt,
        u,
        v,
        w,
        uc,
        vc,
        ua,
        va,
        divgd,
        mfx,
        mfy,
        cx,
        cy,
        crx,
        cry,
        xfx,
        yfx,
        q_con,
        zh,
        heat_source,
        diss_est,
        dt,
    ):
        """D-Grid Shallow Water Routine
        Peforms a full-timestep advance of the D-grid winds and other
        prognostic variables using Lagrangian dynamics on the cubed-sphere.
        described by Lin 1997, Lin 2004 and Harris 2013.
        Args:
            delpc: C-grid  vertical delta in pressure (in)
            delp: D-grid vertical delta in pressure (inout),
            ptc: C-grid potential temperature (in)
            pt: D-grid potnetial teperature (inout)
            u: D-grid x-velocity (inout)
            v: D-grid y-velocity (inout)
            w: vertical velocity (inout)
            uc: C-grid x-velocity (in)
            vc: C-grid y-velocity (in)
            ua: A-grid x-velocity (in)
            va A-grid y-velocity(in)
            divgd: D-grid horizontal divergence (inout)
            mfx: accumulated x mass flux (inout)
            mfy: accumulated y mass flux (inout)
            cx: accumulated Courant number in the x direction (inout)
            cy: accumulated Courant number in the y direction (inout)
            crx: local courant number in the x direction (inout)
            cry: local courant number in the y direction (inout)
            xfx: flux of area in x-direction, in units of m^2 (in)
            yfx: flux of area in y-direction, in units of m^2 (in)
            q_con: total condensate mixing ratio (inout)
            zh: geopotential height defined on layer interfaces (in)
            heat_source:  accumulated heat source (inout)
            diss_est: dissipation estimate (inout)
            dt: acoustic timestep in seconds (in)
        """
        # uc_contra/vc_contra are ut/vt in the original Fortran
        # TODO: when these stencils can be merged investigate whether
        # we can refactor fv_prep into two separate function calls,
        # the chain looks something like:
        #   uc_contra, vc_contra = f(uc, vc, ...)
        #   xfx, yfx = g(uc_contra, vc_contra, ...)
        self.fv_prep(uc, vc, crx, cry, xfx, yfx, self._uc_contra, self._vc_contra, dt)

        self.fvtp2d_dp(
            self._copy_corners(delp),
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_fx,
            self._tmp_fy,
        )

        self._flux_capacitor_stencil(
            cx, cy, mfx, mfy, crx, cry, self._tmp_fx, self._tmp_fy
        )

        if not self.hydrostatic:

            self.delnflux_nosg_w(
                w,
                self._tmp_fx2,
                self._tmp_fy2,
                self._delnflux_damp_w,
                self._tmp_wk,
            )

            self._heat_diss_stencil(
                self._tmp_fx2,
                self._tmp_fy2,
                w,
                self.grid_data.rarea,
                self._tmp_heat_s,
                diss_est,
                self._tmp_dw,
                self._column_namelist["damp_w"],
                self._column_namelist["ke_bg"],
                dt,
            )

            self.fvtp2d_vt_nodelnflux(
                self._copy_corners(w),
                crx,
                cry,
                xfx,
                yfx,
                self._tmp_gx,
                self._tmp_gy,
                x_mass_flux=self._tmp_fx,
                y_mass_flux=self._tmp_fy,
            )

            self._flux_adjust_stencil(
                w,
                delp,
                self._tmp_gx,
                self._tmp_gy,
                self.grid_data.rarea,
            )
        # Fortran: #ifdef USE_COND
        self.fvtp2d_dp_t(
            self._copy_corners(q_con),
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_gx,
            self._tmp_gy,
            mass=delp,
            x_mass_flux=self._tmp_fx,
            y_mass_flux=self._tmp_fy,
        )

        self._flux_adjust_stencil(
            q_con, delp, self._tmp_gx, self._tmp_gy, self.grid_data.rarea
        )

        # Fortran #endif //USE_COND
        self.fvtp2d_tm(
            self._copy_corners(pt),
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_gx,
            self._tmp_gy,
            mass=delp,
            x_mass_flux=self._tmp_fx,
            y_mass_flux=self._tmp_fy,
        )

        self._apply_pt_delp_fluxes(
            gx=self._tmp_gx,
            gy=self._tmp_gy,
            rarea=self.grid_data.rarea,
            fx=self._tmp_fx,
            fy=self._tmp_fy,
            pt=pt,
            delp=delp,
        )
        self._kinetic_energy_update_part_1(
            vc=vc,
            uc=uc,
            cosa=self.grid_data.cosa,
            rsina=self.grid_data.rsina,
            v=v,
            vc_contra=self._vc_contra,
            u=u,
            uc_contra=self._uc_contra,
            dx=self.grid_data.dx,
            dxa=self.grid_data.dxa,
            rdx=self.grid_data.rdx,
            dy=self.grid_data.dy,
            dya=self.grid_data.dya,
            rdy=self.grid_data.rdy,
            ub_contra=self._ub_contra,
            vb_contra=self._vb_contra,
            advected_u=self._advected_u,
            advected_v=self._advected_v,
            dt=dt,
        )
        self._kinetic_energy_update_part_2(
            ub_contra=self._ub_contra,
            advected_u=self._advected_u,
            vb_contra=self._vb_contra,
            advected_v=self._advected_v,
            v=v,
            vc_contra=self._vc_contra,
            u=u,
            uc_contra=self._uc_contra,
            ke=self._tmp_ke,
            dt=dt,
        )

        self._compute_vorticity_stencil(
            u,
            v,
            self.grid_data.dx,
            self.grid_data.dy,
            self.grid_data.rarea,
            self._tmp_wk,
        )

        # TODO if namelist.d_f3d and ROT3 unimplemented
        self._adjust_w_and_qcon_stencil(
            w, delp, self._tmp_dw, q_con, self._column_namelist["damp_w"]
        )
        self.divergence_damping(
            u,
            v,
            va,
            ptc,
            self._tmp_vort,
            ua,
            divgd,
            vc,
            uc,
            delpc,
            self._tmp_ke,
            self._tmp_wk,
            dt,
        )

        self._vort_differencing_stencil(
            self._tmp_vort,
            self._vort_x_delta,
            self._vort_y_delta,
            self._column_namelist["d_con"],
        )

        # Vorticity transport
        self._compute_vort_stencil(self._tmp_wk, self._f0, zh, self._tmp_vort)

        self.fvtp2d_vt_nodelnflux(
            self._copy_corners(self._tmp_vort),
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_fx,
            self._tmp_fy,
        )

        # TODO: what is ke here? It isn't kinetic energy. What are the u and v outputs?
        # they have units of speed * distance, so they are no longer x and y-wind
        # unless before this point u has units of speed divided by distance
        # and is not the x-wind?
        self._u_and_v_from_ke_stencil(
            self._tmp_ke,
            self._tmp_fx,
            self._tmp_fy,
            u,
            v,
            self.grid_data.dx,
            self.grid_data.dy,
        )

        self.delnflux_nosg_v(
            self._tmp_wk,
            self._tmp_ut,
            self._tmp_vt,
            self._delnflux_damp_vt,
            self._tmp_vort,
        )

        # TODO(eddied): These stencils were split to ensure GTC verification
        self._heat_source_from_vorticity_damping_stencil(
            self._vort_x_delta,
            self._vort_y_delta,
            self._tmp_ut,
            self._tmp_vt,
            u,
            v,
            delp,
            self.grid_data.rsin2,
            self.grid_data.cosa_s,
            self.grid_data.rdx,
            self.grid_data.rdy,
            self._tmp_heat_s,
            heat_source,
            diss_est,
            self._column_namelist["d_con"],
        )

        self._update_u_and_v_stencil(
            self._tmp_ut,
            self._tmp_vt,
            u,
            v,
            self._column_namelist["damp_vt"],
        )
