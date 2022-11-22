from typing import Dict, Mapping

import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import pace.fv3core.stencils.delnflux as delnflux
import pace.util
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.fv3core._config import DGridShallowWaterLagrangianDynamicsConfig
from pace.fv3core.stencils.d2a2c_vect import contravariant
from pace.fv3core.stencils.delnflux import DelnFluxNoSG
from pace.fv3core.stencils.divergence_damping import DivergenceDamping
from pace.fv3core.stencils.fvtp2d import FiniteVolumeTransport
from pace.fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from pace.fv3core.stencils.xtp_u import advect_u_along_x
from pace.fv3core.stencils.ytp_v import advect_v_along_y
from pace.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from pace.util.grid import DampingCoefficients, GridData


dcon_threshold = 1e-5


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
        cx (inout): accumulated courant number in the x direction
        cy (inout): accumulated courant number in the y direction
        xflux (inout): flux capacitor in the x direction, accumlated mass flux
        yflux (inout): flux capacitor in the y direction, accumlated mass flux
        crx_adv (in): local courant numver, dt*ut/dx
        cry_adv (in): local courant number dt*vt/dy
        fx (in): 1-D x-direction flux
        fy (in): 1-D y-direction flux
    """
    with computation(PARALLEL), interval(...):
        cx = cx + crx_adv
        cy = cy + cry_adv
        xflux = xflux + fx
        yflux = yflux + fy


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
    """
    Calculate heat generation due to loss of kinetic energy
    due to damping of vertical wind.

    Does nothing for levels where damp_w <= 1e-5.

    diss_est used for stochastic physics only

    Args:
        fx2 (in): x-flux of vertical wind to apply damping
        fy2 (in): y-flux of vertical wind to apply damping
        w (in): vertical wind
        rarea (in):
        heat_source (out): heat source due to loss of kinetic energy
        diss_est (out): heat source due to loss of kinetic energy
        dw (inout): diffusive update to vertical wind, does not
            include convergence correction, since damping is numerical
            and not physical it's not necessary to include.
        damp_w (in):
        ke_bg (in):
    """
    with computation(PARALLEL), interval(...):
        heat_source = 0.0
        diss_est = 0.0
        if damp_w > 1e-5:
            dd8 = ke_bg * abs(dt)
            dw = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
            heat_source = dd8 - dw * (w + 0.5 * dw)
            diss_est = heat_source


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


def apply_fluxes(
    q: FloatField, delp: FloatField, gx: FloatField, gy: FloatField, rarea: FloatFieldIJ
):
    """
    Update q according to fluxes gx and gy.

    Changes the units of q to be pressure-weighted.

    Args:
        q (inout): any scalar, the fluxes gx and gy are applied and the output is
            mass-weighted (replaced with units of q * delp)
        delp (in): pressure thickness of layer
        gx (in): x-flux of q in units of q * Pa * area
        gy (in): y-flux of q in units of q * Pa * area
        rarea (in): 1 / area
    """
    # TODO: this function changes the units and therefore meaning of q,
    # is there any way we can avoid doing so?
    # the next time w and q_con (passed as q to this routine) are used
    # is in adjust_w_and_qcon, where they are divided by an updated delp to return
    # to the original units.
    with computation(PARALLEL), interval(...):
        # in the original Fortran, this uses `w` instead of `q`
        q = q * delp + flux_increment(gx, gy, rarea)


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
    """
    Args:
        fx (in):
        fy (in):
        pt (inout):
        delp (inout):
        gx (in):
        gy (in):
        rarea (in):
    """
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
    """
    Args:
        fx (in):
        fy (in):
        pt (inout):
        delp (inout):
        gx (in):
        gy (in):
        rarea (in):
    """
    with computation(PARALLEL), interval(...):
        pt, delp = apply_pt_delp_fluxes(gx, gy, rarea, fx, fy, pt, delp)


def compute_kinetic_energy(
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
    dt_kinetic_energy_on_cell_corners: FloatField,
    dt: float,
):
    """
    Args:
        vc (in): y-wind on c-grid
        uc (in): x-wind on c-grid
        cosa (in):
        rsina (in):
        v (in): y-wind on d-grid
        vc_contra (in): contravariant y-wind on c-grid
        u (in): x-wind on d-grid
        uc_contra (in): contravariant x-wind on c-grid
        dx (in):
        dxa (in):
        rdx (in):
        dy (in):
        dya (in):
        rdy (in):
        dt_kinetic_energy_on_cell_corners (out): kinetic energy on cell corners,
            as defined in FV3 documentation by equation 6.3, multiplied by dt
        dt: timestep
    """
    with computation(PARALLEL), interval(...):
        ub_contra, vb_contra = interpolate_uc_vc_to_cell_corners(
            uc, vc, cosa, rsina, uc_contra, vc_contra
        )
        advected_v = advect_v_along_y(v, vb_contra, rdy=rdy, dy=dy, dya=dya, dt=dt)
        advected_u = advect_u_along_x(u, ub_contra, rdx=rdx, dx=dx, dxa=dxa, dt=dt)
        # makes sure the kinetic energy part of the governing equation is computed
        # the same way as the vorticity flux part (in terms of time splitting)
        # to avoid a Hollingsworth-Kallberg instability
        dt_kinetic_energy_on_cell_corners = (
            0.5 * dt * (ub_contra * advected_u + vb_contra * advected_v)
        )
        dt_kinetic_energy_on_cell_corners = all_corners_ke(
            dt_kinetic_energy_on_cell_corners, u, v, uc_contra, vc_contra, dt
        )


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


def compute_vorticity(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    rarea: FloatFieldIJ,
    vorticity: FloatField,
):
    """
    Compute cell mean vorticity using Stokes' theorem (using average circulation).

    Args:
        u (in):
        v (in):
        dx (in):
        dy (in):
        rarea (in):
        vorticity (out):
    """
    with computation(PARALLEL), interval(...):
        # TODO: ask Lucas why vorticity is computed with this particular treatment
        # of dx, dy, and rarea. The original code read like:
        #     u_dx = u * dx
        #     v_dy = v * dy
        #     vorticity = rarea * (u_dx - u_dx[0, 1, 0] - v_dy + v_dy[1, 0, 0])
        # cell-mean vorticity is equal to the circulation around the gridcell
        # divided by the area of the gridcell. It isn't exactly true that
        # area = dx * dy, so the form below is necessary to get an exact result.
        rdy_tmp = rarea * dx
        rdx_tmp = rarea * dy
        vorticity = (u - u[0, 1, 0] * dx[0, 1] / dx) * rdy_tmp + (
            v[1, 0, 0] * dy[1, 0] / dy - v
        ) * rdx_tmp


def adjust_w_and_qcon(
    w: FloatField,
    delp: FloatField,
    dw: FloatField,
    q_con: FloatField,
    damp_w: FloatFieldK,
):
    """
    Args:
        w (inout):
        delp (in):
        dw (in):
        q_con (inout):
        damp_w (in):
    """
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = w + dw if damp_w > 1e-5 else w
        # Fortran: #ifdef USE_COND
        q_con = q_con / delp


def vort_differencing(
    vort: FloatField,
    vort_x_delta: FloatField,
    vort_y_delta: FloatField,
    dcon: FloatFieldK,
):
    """
    Intermediate operation for computing heating due to diffusion

    Args:
        vort (in):
        vort_x_delta (out):
        vort_y_delta (out):
        dcon (in):
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        # TODO: this should likely be dcon[k] rather than dcon[0] so that this
        # can be turned on and off per-layer
        if dcon[0] > dcon_threshold:
            # Creating a gtscript function for the ub/vb computation
            # results in an "NotImplementedError" error for Jenkins
            # Inlining the ub/vb computation in this stencil resolves the Jenkins error
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                vort_x_delta = vort - vort[1, 0, 0]
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                vort_y_delta = vort - vort[0, 1, 0]


# TODO: This is untested and the radius may be incorrect
@gtscript.function
def coriolis_force_correction(zh, radius):
    return 1.0 + (zh + zh[0, 0, 1]) / radius


def rel_vorticity_to_abs(
    relative_vorticity: FloatField,
    f0: FloatFieldIJ,
    absolute_vorticity: FloatField,
):
    """
    Args:
        relative_vorticity (in): a-grid relative vorticity computed before
            divergence damping
        f0 (in): planetary vorticity
        absolute_vorticity (out): absolute vorticity on cell center (a-grid)
    """
    with computation(PARALLEL), interval(...):
        absolute_vorticity = relative_vorticity + f0


@gtscript.function
def u_from_ke(ke, u, dx, fy):
    """
    Described in section 5.2 eq 5.3d and 5.3e of FV3 docs.

    Can re-write the velocity equation in a vector-invariant form
    where no derivatives of velocity are taken:

    du/dt = (y-flux of absolute vorticity) - d/dx(KE) + (pressure gradient term)
    dv/dt = -(x-flux of absolute vorticity) - d/dy(KE) + (pressure gradient term)

    Equations 6.1d and 6.1e are the discrete form of the above equations.

    The pressure gradient term is added later, not here.

    This is important because then you don't need to worry about how
    the coordinate vectors are being advected, all advected terms
    are scalar.

    Args:
        ke (in): dt * kinetic energy defined on cell corners
        u (in): x-velocity
        dx (in): grid cell width
        fy (in): flux of absolute vorticity in y-direction
    """
    return u * dx + ke - ke[1, 0, 0] + fy


@gtscript.function
def v_from_ke(ke, v, dy, fx):
    # see docstring for u_from_ke
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
    """
    Update the u and v velocities from the kinetic energy and
    flux of absolute vorticity.

    Does not include the pressure gradient force update to the winds.

    Converts u and v to u*dx and v*dy.

    Args:
        ke (in): dt*kinetic_energy defined on cell corners
        fx (in): flux of absolute vorticity in x-direction
        fy (in): flux of absolute vorticity in y-direction
        u (inout): On input, u. On output, u * dx
        v (inout): On input, v. On output, v * dy
        dx (in):
        dy (in):
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    # TODO: this function does not return u and v, it returns something
    # like u * dx and v * dy. Rename this function and its inouts.
    # rename u/v to u_dx and v_dy

    with computation(PARALLEL), interval(...):
        # TODO: may be able to remove local regions once this stencil and
        # heat_from_damping are in the same stencil
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            u = u_from_ke(ke, u, dx, fy)
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            v = v_from_ke(ke, v, dy, fx)


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return (
        rsin2
        * 0.25
        * (
            (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
            + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
            - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
        )
    )


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

    Described in Section 8.5 of FV3 docs.

    Args:
        vort_x_delta (in): x finite difference of damped b-grid vorticity
        vort_y_delta (in): y finite difference of damped b-grid vorticity
        ut (in): damping flux in x-direction of a-grid relative vorticity
        vt (in): damping flux in y-direction of a-grid relative vorticity
        u (in): d-grid x-velocity
        v (in): d-grid y-velocity
        delp (in): pressure thickness of layer
        rsin2 (in):
        cosa_s (in):
        rdx (in): 1 / dx
        rdy (in): 1 / dy
        heat_source (inout): heat source from vorticity damping
            implied by energy conservation
        heat_source_total (inout): accumulated heat source
        dissipation_estimate (out): dissipation estimate, only calculated if
            calculate_dissipation_estimate is 1. Used for stochastic kinetic
            energy backscatter (skeb) routine.
        kinetic_energy_fraction_to_damp (in): the fraction of kinetic energy
            to explicitly damp and convert into heat.
    """
    from __externals__ import (
        d_con,
        do_stochastic_ke_backscatter,
        local_ie,
        local_is,
        local_je,
        local_js,
    )

    # TODO: rename d_con in the namelist itself
    # d_con is portion of dissipative kinetic energy restored as heat, turns off
    # if it's below some threshold
    # TODO: d_con is passed both as an external ad as an argument
    # (as kinetic_energy_fraction_to_damp), clean this up

    with computation(PARALLEL), interval(...):
        ubt = (vort_x_delta + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vort_y_delta - ut) * rdy
        fx = v * rdy
        gx = fx * vbt

        if (
            kinetic_energy_fraction_to_damp > dcon_threshold
        ) or do_stochastic_ke_backscatter:
            u2 = fy + fy[0, 1, 0]
            du2 = ubt + ubt[0, 1, 0]
            v2 = fx + fx[1, 0, 0]
            dv2 = vbt + vbt[1, 0, 0]
            dampterm = heat_damping_term(
                ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2
            )
            heat_source = delp * (
                heat_source - kinetic_energy_fraction_to_damp * dampterm
            )

        if __INLINED((d_con > dcon_threshold) or do_stochastic_ke_backscatter):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
                heat_source_total = heat_source_total + heat_source
                if __INLINED(do_stochastic_ke_backscatter):
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
    Updates u and v with fluxes resulting from the diffusive flux of vorticity.

    See docstring of u_from_ke for description of equations.

    Args:
        ut (in): x-direction diffusivie vorticity flux
        vt (in): y-direction diffusivie vorticity flux
        u (inout): d-grid x-velocity
        v (inout): d-grid y-velocity
        damp_vt (in): column scalar for damping vorticity
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        if damp_vt > 1e-5:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                u += vt
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                v -= ut


# Set the unique parameters for the smallest
# k-values, e.g. k = 0, 1, 2 when generating
# the column namelist
def set_low_kvals(col: Mapping[str, pace.util.Quantity], k):
    for name in ["nord", "nord_w", "d_con"]:
        col[name].view[k] = 0
    col["damp_w"].view[k] = col["d2_divg"].view[k]


# For the column namelist at a specific k-level
# set the vorticity parameters if do_vort_damp is true
def vorticity_damping_option(column, k, do_vort_damp):
    if do_vort_damp:
        column["nord_v"].view[k] = 0
        column["damp_vt"].view[k] = 0.5 * column["d2_divg"].view[k]


def lowest_kvals(column, k, do_vort_damp):
    set_low_kvals(column, k)
    vorticity_damping_option(column, k, do_vort_damp)


def get_column_namelist(
    config: DGridShallowWaterLagrangianDynamicsConfig,
    quantity_factory: pace.util.QuantityFactory,
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
    col: Dict[str, pace.util.Quantity] = {}
    for name in all_names:
        # TODO: fill units information
        col[name] = quantity_factory.zeros(dims=[Z_DIM], units="unknown")
    for name in direct_namelist:
        col[name].view[:] = getattr(config, name)

    col["d2_divg"].view[:] = min(0.2, config.d2_bg)
    col["nord_v"].view[:] = min(2, col["nord"].view[0])
    col["nord_w"].view[:] = col["nord_v"].view[0]
    col["nord_t"].view[:] = col["nord_v"].view[0]
    if config.do_vort_damp:
        col["damp_vt"].view[:] = config.vtdm4
    else:
        col["damp_vt"].view[:] = 0
    col["damp_w"].view[:] = col["damp_vt"].view[0]
    col["damp_t"].view[:] = col["damp_vt"].view[0]
    if (
        col["d2_divg"].extent[col["d2_divg"].dims.index(Z_DIM)] == 1
        or config.n_sponge < 0
    ):
        col["d2_divg"].view[0] = config.d2_bg
    else:
        col["d2_divg"].view[0] = max(0.01, config.d2_bg, config.d2_bg_k1)
        lowest_kvals(col, 0, config.do_vort_damp)
        if config.d2_bg_k2 > 0.01:
            col["d2_divg"].view[1] = max(config.d2_bg, config.d2_bg_k2)
            lowest_kvals(col, 1, config.do_vort_damp)
        if config.d2_bg_k2 > 0.05:
            col["d2_divg"].view[2] = max(config.d2_bg, 0.2 * config.d2_bg_k2)
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


class DGridShallowWaterLagrangianDynamics:
    """
    Fortran name is the d_sw subroutine
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        column_namelist,
        nested: bool,
        stretched_grid: bool,
        config: DGridShallowWaterLagrangianDynamicsConfig,
    ):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.grid_data = grid_data
        self._f0 = self.grid_data.fC_agrid

        self.grid_indexing = stencil_factory.grid_indexing
        assert config.grid_type < 3, "ubke and vbke only implemented for grid_type < 3"
        assert not config.inline_q, "inline_q not yet implemented"
        assert (
            config.d_ext <= 0
        ), "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        assert (column_namelist["damp_vt"].view[:] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_vt > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true
        assert (column_namelist["damp_w"].view[:] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_w > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true

        # only compute for k-levels where this is true
        self.hydrostatic = config.hydrostatic

        def make_quantity():
            return quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._tmp_heat_s = make_quantity()
        self._vort_x_delta = make_quantity()
        self._vort_y_delta = make_quantity()
        self._dt_kinetic_energy_on_cell_corners = make_quantity()
        self._abs_vorticity_agrid = make_quantity()
        self._uc_contra = make_quantity()
        self._vc_contra = make_quantity()
        self._tmp_ut = make_quantity()
        self._tmp_vt = make_quantity()
        self._tmp_fx = make_quantity()
        self._tmp_fy = make_quantity()
        self._tmp_gx = make_quantity()
        self._tmp_gy = make_quantity()
        self._tmp_dw = make_quantity()
        self._tmp_wk = make_quantity()
        self._vorticity_agrid = make_quantity()
        self._vorticity_bgrid_damped = make_quantity()
        self._tmp_fx2 = make_quantity()
        self._tmp_fy2 = make_quantity()
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
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_dp,
            nord=self._column_namelist["nord_v"],
            damp_c=self._column_namelist["damp_vt"],
        )
        self.fvtp2d_dp_t = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_dp,
            nord=self._column_namelist["nord_t"],
            damp_c=self._column_namelist["damp_t"],
        )
        self.fvtp2d_tm = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_tm,
            nord=self._column_namelist["nord_v"],
            damp_c=self._column_namelist["damp_vt"],
        )
        self.fvtp2d_vt_nodelnflux = FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
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
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            nested=nested,
            stretched_grid=stretched_grid,
            dddmp=config.dddmp,
            d4_bg=config.d4_bg,
            nord=config.nord,
            grid_type=config.grid_type,
            nord_col=column_namelist["nord"],
            d2_bg=column_namelist["d2_divg"],
        )

        self._apply_pt_delp_fluxes = stencil_factory.from_dims_halo(
            func=apply_pt_delp_fluxes_stencil_defn,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "inline_q": config.inline_q,
            },
        )
        self._compute_kinetic_energy = stencil_factory.from_dims_halo(
            func=compute_kinetic_energy,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            externals={
                "iord": config.hord_mt,
                "jord": config.hord_mt,
                "mord": config.hord_mt,
                "xt_minmax": False,
                "yt_minmax": False,
            },
        )
        self._apply_fluxes = stencil_factory.from_dims_halo(
            func=apply_fluxes, compute_dims=[X_DIM, Y_DIM, Z_DIM]
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

        if config.do_f3d:
            raise NotImplementedError("do_f3d is not implemented")

        self._rel_vorticity_to_abs = stencil_factory.from_dims_halo(
            func=rel_vorticity_to_abs,
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
                    "do_stochastic_ke_backscatter": config.do_skeb,
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
        self._delnflux_damp_vt = delnflux.calc_damp(
            damp_c=self._column_namelist["damp_vt"],
            da_min=damping_coefficients.da_min_c,
            nord=self._column_namelist["nord_v"],
        )
        self._delnflux_damp_w = delnflux.calc_damp(
            damp_c=self._column_namelist["damp_w"],
            da_min=damping_coefficients.da_min_c,
            nord=self._column_namelist["nord_w"],
        )

    def __call__(
        self,
        delpc,
        delp,
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
        """
        D-Grid shallow water routine, peforms a full-timestep advance
        of the D-grid winds and other prognostic variables using Lagrangian
        dynamics on the cubed-sphere.

        Described by Lin 1997, Lin 2004 and Harris 2013.

        Args:
            delpc (inout): C-grid  vertical delta in pressure
            delp (inout): D-grid vertical delta in pressure
            pt (inout): D-grid potential teperature
            u (inout): D-grid x-velocity
            v (inout): D-grid y-velocity
            w (inout): vertical velocity
            uc (in): C-grid x-velocity
            vc (in): C-grid y-velocity
            ua (in): A-grid x-velocity
            va (in) A-grid y-velocity
            divgd (inout): D-grid horizontal divergence, used for divergence
                damping. TODO: Variable gets re-used for other purposes causing
                it to be inout, can we refactor those?
                corner handling, higher-order divergence damping temporary
            mfx (inout): accumulated x mass flux
            mfy (inout): accumulated y mass flux
            cx (inout): accumulated Courant number in the x direction
            cy (inout): accumulated Courant number in the y direction
            crx (out): local courant number in the x direction
            cry (out): local courant number in the y direction
            xfx (out): flux of area in x-direction, in units of m^2
            yfx (out): flux of area in y-direction, in units of m^2
            q_con (inout): total condensate mixing ratio,
                used for ideal gas law calculation
            zh (in): geopotential height defined on layer interfaces
            heat_source (inout):  accumulated heat source
            diss_est (inout): dissipation estimate
            dt (in): acoustic timestep in seconds
        """
        # uc_contra/vc_contra are ut/vt in the original Fortran
        # TODO: when these stencils can be merged investigate whether
        # we can refactor fv_prep into two separate function calls,
        # the chain looks something like:
        #   uc_contra, vc_contra = f(uc, vc, ...)
        #   xfx, yfx = g(uc_contra, vc_contra, ...)

        # TODO: ptc may only be used as a temporary under this scope, investigate
        # and decouple it from higher level if possible (i.e. if not a real output)
        self.fv_prep(uc, vc, crx, cry, xfx, yfx, self._uc_contra, self._vc_contra, dt)

        # TODO: the structure of much of this is to get fluxes from fvtp2d and then
        # apply them in various similar stencils - can these steps be merged?

        # [DaCe] Remove CopiedCorners
        self.fvtp2d_dp(
            delp,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_fx,
            self._tmp_fy,
        )

        # TODO: part of flux_capacitor_stencil (updating cx, cy)
        # should be mergeable with fv_prep, the other part (updating xflux, yflux)
        # should be mergeable with any compute domain stencil in this object

        # fluxes are accumulated over substeps so we can later apply them in
        # tracer advection
        self._flux_capacitor_stencil(
            cx, cy, mfx, mfy, crx, cry, self._tmp_fx, self._tmp_fy
        )

        # TODO: output value for tmp_wk here is never used, refactor so it is
        # not unnecessarily computed
        # compute diffusive flux for veritcal velocity
        self.delnflux_nosg_w(
            w,
            self._tmp_fx2,
            self._tmp_fy2,
            self._delnflux_damp_w,
            self._tmp_wk,
        )
        # gets the heat tendency due to w dissipation, and the
        # w update implied by the fluxes
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

        # compute and apply advective fluxes for w and q_con so that we can
        # apply them at the same time as we apply w dissipative fluxes
        self.fvtp2d_vt_nodelnflux(
            w,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_gx,
            self._tmp_gy,
            x_mass_flux=self._tmp_fx,
            y_mass_flux=self._tmp_fy,
        )

        self._apply_fluxes(
            w,
            delp,
            self._tmp_gx,
            self._tmp_gy,
            self.grid_data.rarea,
        )
        # Fortran: #ifdef USE_COND
        # [DaCe] Remove CopiedCorners
        self.fvtp2d_dp_t(
            q_con,
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

        self._apply_fluxes(
            q_con, delp, self._tmp_gx, self._tmp_gy, self.grid_data.rarea
        )

        # Fortran #endif //USE_COND

        # [DaCe] Remove CopiedCorners
        self.fvtp2d_tm(
            pt,
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

        self._adjust_w_and_qcon_stencil(
            w, delp, self._tmp_dw, q_con, self._column_namelist["damp_w"]
        )
        # at this point, pt, delp, w and q_con have been stepped forward in time
        # the rest of this function updates the winds
        self._compute_kinetic_energy(
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
            dt_kinetic_energy_on_cell_corners=self._dt_kinetic_energy_on_cell_corners,
            dt=dt,
        )

        self._compute_vorticity_stencil(
            u,
            v,
            self.grid_data.dx,
            self.grid_data.dy,
            self.grid_data.rarea,
            self._vorticity_agrid,
        )

        self.divergence_damping(
            u,
            v,
            va,
            self._vorticity_bgrid_damped,
            ua,
            divgd,
            vc,
            uc,
            delpc,
            self._dt_kinetic_energy_on_cell_corners,
            # a-grid relative vorticity computed before divergence damping
            self._vorticity_agrid,
            dt,
        )

        # Vorticity transport
        self._rel_vorticity_to_abs(
            self._vorticity_agrid, self._f0, self._abs_vorticity_agrid
        )

        # [DaCe] Unroll CopiedCorners see __init__
        self.fvtp2d_vt_nodelnflux(
            self._abs_vorticity_agrid,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_fx,  # flux of absolute vorticity
            self._tmp_fy,
        )

        self._u_and_v_from_ke_stencil(
            self._dt_kinetic_energy_on_cell_corners,
            self._tmp_fx,
            self._tmp_fy,
            u,
            v,
            self.grid_data.dx,
            self.grid_data.dy,
        )

        # TODO: use a separate temporary/storage for this variable name
        damped_rel_vorticity_agrid = self._abs_vorticity_agrid

        self.delnflux_nosg_v(
            self._vorticity_agrid,
            self._tmp_ut,
            self._tmp_vt,
            self._delnflux_damp_vt,
            damped_rel_vorticity_agrid,
        )
        # TODO(eddied): These stencils were split to ensure GTC verification
        self._vort_differencing_stencil(
            self._vorticity_bgrid_damped,
            self._vort_x_delta,
            self._vort_y_delta,
            self._column_namelist["d_con"],
        )
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
