import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    external_assert,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.divergence_damping as divdamp
import fv3core.stencils.fxadv as fxadv
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.stencils.xtp_u import XTP_U
from fv3core.stencils.ytp_v import YTP_V
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


dcon_threshold = 1e-5


def grid():
    return spec.grid


def k_bounds():
    # UpdatedzD needs to go one k level higher than D_SW, to the buffer point that
    # usually isn't used. To reuse the same 'column_namelist' and remove the
    # specification of 'kstart' and 'nk in many methods, we just make all of the
    # column namelist calculations go to the top of the array
    return [[0, 1], [1, 1], [2, 1], [3, grid().npz - 2]]


@gtscript.function
def flux_component(gx, gy, rarea):
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


@gtscript.function
def flux_integral(w, delp, gx, gy, rarea):
    return w * delp + flux_component(gx, gy, rarea)


@gtstencil()
def flux_adjust(
    w: FloatField, delp: FloatField, gx: FloatField, gy: FloatField, rarea: FloatFieldIJ
):
    with computation(PARALLEL), interval(...):
        w = flux_integral(w, delp, gx, gy, rarea)


@gtstencil()
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
def horizontal_relative_vorticity_from_winds(u, v, ut, vt, dx, dy, rarea, vorticity):
    """
    Compute the area mean relative vorticity in the z-direction from the D-grid winds.

    Args:
        u (in): x-direction wind on D grid
        v (in): y-direction wind on D grid
        ut (out): u * dx
        vt (out): v * dy
        dx (in): gridcell width in x-direction
        dy (in): gridcell width in y-direction
        rarea (in): inverse of area
        vorticity (out): area mean horizontal relative vorticity
    """

    vt = u * dx
    ut = v * dy
    vorticity = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])

    return vt, ut, vorticity


@gtscript.function
def all_corners_ke(ke, u, v, ut, vt, dt):
    from __externals__ import i_end, i_start, j_end, j_start

    # Assumption: not __INLINED(spec.grid.nested)
    with horizontal(region[i_start, j_start]):
        ke = corners.corner_ke(ke, u, v, ut, vt, dt, 0, 0, -1, 1)
    with horizontal(region[i_end + 1, j_start]):
        ke = corners.corner_ke(ke, u, v, ut, vt, dt, -1, 0, 0, -1)
    with horizontal(region[i_end + 1, j_end + 1]):
        ke = corners.corner_ke(ke, u, v, ut, vt, dt, -1, -1, 0, 1)
    with horizontal(region[i_start, j_end + 1]):
        ke = corners.corner_ke(ke, u, v, ut, vt, dt, 0, -1, -1, -1)

    return ke


@gtstencil()
def not_inlineq_pressure(
    gx: FloatField,
    gy: FloatField,
    rarea: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    pt: FloatField,
    delp: FloatField,
):
    with computation(PARALLEL), interval(...):
        pt = flux_integral(pt, delp, gx, gy, rarea)
        delp = delp + flux_component(fx, fy, rarea)
        pt = pt / delp


@gtstencil()
def not_inlineq_pressure_and_vbke(
    gx: FloatField,
    gy: FloatField,
    rarea: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    pt: FloatField,
    delp: FloatField,
    vc: FloatField,
    uc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    vt: FloatField,
    vb: FloatField,
    dt4: float,
    dt5: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        # TODO: only needed for d_sw validation
        if __INLINED(spec.namelist.inline_q == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
                pt = flux_integral(pt, delp, gx, gy, rarea)
                delp = delp + flux_component(fx, fy, rarea)
                pt = pt / delp
        external_assert(spec.namelist.grid_type < 3)
        vb = vbke(vc, uc, cosa, rsina, vt, vb, dt4, dt5)


@gtscript.function
def ke_from_bwind(ke, ub, vb):
    return 0.5 * (ke + ub * vb)


@gtstencil()
def ub_vb_from_vort(
    vort: FloatField,
    ub: FloatField,
    vb: FloatField,
    dcon: FloatFieldK,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        if dcon[0] > dcon_threshold:
            # Creating a gtscript function for the ub/vb computation
            # results in an "NotImplementedError" error for Jenkins
            # Inlining the ub/vb computation in this stencil resolves the Jenkins error
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                ub = vort - vort[1, 0, 0]
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                vb = vort - vort[0, 1, 0]


@gtscript.function
def u_from_ke(ke, vt, fy):
    return vt + ke - ke[1, 0, 0] + fy


@gtscript.function
def v_from_ke(ke, ut, fx):
    return ut + ke - ke[0, 1, 0] - fx


@gtstencil()
def u_and_v_from_ke(
    ke: FloatField,
    ut: FloatField,
    vt: FloatField,
    fx: FloatField,
    fy: FloatField,
    u: FloatField,
    v: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        # TODO: may be able to remove local regions once this stencil and
        # heat_from_damping are in the same stencil
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            u = u_from_ke(ke, vt, fy)
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            v = v_from_ke(ke, ut, fx)


# TODO: This is untested and the radius may be incorrect
@gtscript.function
def coriolis_force_correction(zh, radius):
    return 1.0 + (zh + zh[0, 0, 1]) / radius


@gtstencil(externals={"radius": constants.RADIUS})
def compute_vorticity(
    wk: FloatField,
    f0: FloatFieldIJ,
    zh: FloatField,
    vort: FloatField,
):

    from __externals__ import namelist, radius

    with computation(PARALLEL), interval(...):
        if __INLINED(namelist.do_f3d and not namelist.hydrostatic):
            z_rat = coriolis_force_correction(zh, radius)
            vort = wk + f0 * z_rat
        else:
            vort = wk[0, 0, 0] + f0[0, 0]


@gtstencil()
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
        # USE_COND
        q_con = q_con / delp


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


@gtstencil()
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


@gtstencil()
def heat_source_from_vorticity_damping(
    ub: FloatField,
    vb: FloatField,
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
    damp_vt: FloatFieldK,
):
    """
    Calculates heat source from vorticity damping implied by energy conservation.
    Updates u and v
    Args:
        ub (in)
        vb (in)
        ut (in)
        vt (in)
        u (in)
        v (in)
        delp (in)
        rsin2 (in)
        cosa_s (in)
        rdx (in): radius of Earth multiplied by x-direction gridcell width
        rdy (in): radius of Earth multiplied by y-direction gridcell width
        heat_source (out): heat source from vorticity damping
            implied by energy conservation
        heat_source_total: (out) accumulated heat source
        dissipation_estimate (out): dissipation estimate, only calculated if
            calculate_dissipation_estimate is 1
        kinetic_energy_fraction_to_damp (in): according to its comment in fv_arrays,
            the fraction of kinetic energy to explicitly damp and convert into heat.
            TODO: confirm this description is accurate, why is it multiplied
            by 0.25 below?
        damp_vt: column scalar for damping vorticity
    """
    from __externals__ import local_ie, local_is, local_je, local_js, namelist

    with computation(PARALLEL), interval(...):
        # if (kinetic_energy_fraction_to_damp[0] > dcon_threshold) or namelist.do_skeb:
        heat_s = heat_source
        diss_e = dissipation_estimate
        ubt = (ub + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vb - ut) * rdy
        fx = v * rdy
        gx = fx * vbt
    with computation(PARALLEL), interval(...):
        if (kinetic_energy_fraction_to_damp[0] > dcon_threshold) or namelist.do_skeb:
            u2 = fy + fy[0, 1, 0]
            du2 = ubt + ubt[0, 1, 0]
            v2 = fx + fx[1, 0, 0]
            dv2 = vbt + vbt[1, 0, 0]
            dampterm = heat_damping_term(
                ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2
            )
            heat_source = delp * (
                heat_s - 0.25 * kinetic_energy_fraction_to_damp[0] * dampterm
            )
    with computation(PARALLEL), interval(...):
        if __INLINED((namelist.d_con > dcon_threshold) or namelist.do_skeb):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
                heat_source_total = heat_source_total + heat_source
                # do_skeb could be renamed to calculate_dissipation_estimate
                # when d_sw is converted into a D_SW object
                if __INLINED(namelist.do_skeb == 1):
                    dissipation_estimate = diss_e - dampterm
    with computation(PARALLEL), interval(...):
        if damp_vt > 1e-5:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                u = u + vt
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                v = v - ut


@gtstencil()
def ke_horizontal_vorticity(
    ke: FloatField,
    u: FloatField,
    v: FloatField,
    ub: FloatField,
    vb: FloatField,
    ut: FloatField,
    vt: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    rarea: FloatFieldIJ,
    vorticity: FloatField,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        ke = ke_from_bwind(ke, ub, vb)
        ke = all_corners_ke(ke, u, v, ut, vt, dt)
        vt, ut, vorticity = horizontal_relative_vorticity_from_winds(
            u, v, ut, vt, dx, dy, rarea, vorticity
        )


# Set the unique parameters for the smallest
# k-values, e.g. k = 0, 1, 2 when generating
# the column namelist
def set_low_kvals(col, k):
    for name in ["nord", "nord_w", "d_con"]:
        col[name][k] = 0
    col["damp_w"][k] = col["d2_divg"][k]


# For the column namelist at a spcific k-level
# set the vorticity parameters if do_vort_damp is true
def vorticity_damping_option(column, k):
    if spec.namelist.do_vort_damp:
        column["nord_v"][k] = 0
        column["damp_vt"][k] = 0.5 * column["d2_divg"][k]


def lowest_kvals(column, k):
    set_low_kvals(column, k)
    vorticity_damping_option(column, k)


def max_d2_bg0():
    return max(0.01, spec.namelist.d2_bg, spec.namelist.d2_bg_k1)


def max_d2_bg1():
    return max(spec.namelist.d2_bg, spec.namelist.d2_bg_k2)


def get_column_namelist():
    """
    Generate a dictionary of columns that specify how parameters (such as nord, damp)
    used in several functions called by D_SW vary over the k-dimension.

    In a near-future PR, the need for this will disappear as we refactor
    individual modules to apply this parameter variation explicitly in the
    stencils themselves. If it doesn't, we should compute it only in the init phase.
    The unique set of all column parameters is specified by k_bounds. For each k range
    as specified by (kstart, nk) this sets what several different parameters are.
    It previously was a dictionary with the k value as the key, the value being another
    dictionary of values, but this did not work when we removed the k loop from some
    modules and instead wanted to push the whole column ingestion down a level.
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
        col[name] = utils.make_storage_from_shape(
            (spec.grid.npz + 1,), (0,), cache_key="nam-" + name
        )
    for name in direct_namelist:
        col[name][:] = getattr(spec.namelist, name)

    col["d2_divg"][:] = min(0.2, spec.namelist.d2_bg)
    col["nord_v"][:] = min(2, col["nord"][0])
    col["nord_w"][:] = col["nord_v"][0]
    col["nord_t"][:] = col["nord_v"][0]
    if spec.namelist.do_vort_damp:
        col["damp_vt"][:] = spec.namelist.vtdm4
    else:
        col["damp_vt"][:] = 0
    col["damp_w"][:] = col["damp_vt"][0]
    col["damp_t"][:] = col["damp_vt"][0]
    if grid().npz == 1 or spec.namelist.n_sponge < 0:
        col["d2_divg"][0] = spec.namelist.d2_bg
    else:
        col["d2_divg"][0] = max_d2_bg0()
        lowest_kvals(col, 0)
        if spec.namelist.d2_bg_k2 > 0.01:
            col["d2_divg"][1] = max_d2_bg1()
            lowest_kvals(col, 1)
        if spec.namelist.d2_bg_k2 > 0.05:
            col["d2_divg"][2] = max(spec.namelist.d2_bg, 0.2 * spec.namelist.d2_bg_k2)
            set_low_kvals(col, 2)
    return col


def damp_vertical_wind(w, heat_s, diss_est, dt, column_namelist):
    dw = utils.make_storage_from_shape(
        w.shape, grid().compute_origin(), cache_key="d_sw_dw"
    )
    wk = utils.make_storage_from_shape(
        w.shape, grid().full_origin(), cache_key="d_sw_wk"
    )
    fx2 = utils.make_storage_from_shape(
        w.shape, grid().full_origin(), cache_key="d_sw_fx2"
    )
    fy2 = utils.make_storage_from_shape(
        w.shape, grid().full_origin(), cache_key="d_sw_fy2"
    )
    for kstart, nk in k_bounds():
        if column_namelist["damp_w"][kstart] > 1e-5:
            damp4 = (column_namelist["damp_w"][kstart] * grid().da_min_c) ** (
                column_namelist["nord_w"][kstart] + 1
            )
            delnflux.compute_no_sg(
                w,
                fx2,
                fy2,
                column_namelist["nord_w"][kstart],
                damp4,
                wk,
                kstart=kstart,
                nk=nk,
            )
    heat_diss(
        fx2,
        fy2,
        w,
        grid().rarea,
        heat_s,
        diss_est,
        dw,
        column_namelist["damp_w"],
        column_namelist["ke_bg"],
        dt,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(),
    )
    return dw, wk


@gtscript.function
def ubke(uc, vc, cosa, rsina, ut, ub, dt4, dt5):
    from __externals__ import i_end, i_start, j_end, j_start

    ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina
    # if __INLINED(spec.namelist.grid_type < 3):
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        ub = dt5 * (ut[0, -1, 0] + ut)

    return ub


@gtstencil()
def mult_ubke(
    vb: FloatField,
    ke: FloatField,
    uc: FloatField,
    vc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    ut: FloatField,
    ub: FloatField,
    dt4: float,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        ke = vb * ub
        external_assert(spec.namelist.grid_type < 3)
        ub = ubke(uc, vc, cosa, rsina, ut, ub, dt4, dt5)


@gtscript.function
def vbke(vc, uc, cosa, rsina, vt, vb, dt4, dt5):
    from __externals__ import i_end, i_start, j_end, j_start

    vb = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina
    # ASSUME : if __INLINED(spec.namelist.grid_type < 3):
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        vb = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        vb = dt5 * (vt[-1, 0, 0] + vt)

    return vb


def compute(
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
    column_namelist = get_column_namelist()

    if spec.namelist.d_ext > 0:
        raise Exception(
            "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        )
    shape = heat_source.shape
    heat_s = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_heat_s"
    )
    ub = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_ub"
    )
    vb = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_vb"
    )
    ke = utils.make_storage_from_shape(shape, grid().full_origin(), cache_key="d_sw_ke")
    vort = utils.make_storage_from_shape(
        shape, grid().full_origin(), cache_key="d_sw_vort"
    )
    ut = utils.make_storage_from_shape(shape, grid().full_origin(), cache_key="d_sw_ut")
    vt = utils.make_storage_from_shape(shape, grid().full_origin(), cache_key="d_sw_vt")
    fx = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_fx"
    )
    fy = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_fy"
    )
    gx = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_gx"
    )
    gy = utils.make_storage_from_shape(
        shape, grid().compute_origin(), cache_key="d_sw_gy"
    )

    fvtp2d_dp = utils.cached_stencil_class(FiniteVolumeTransport)(
        spec.namelist, spec.namelist.hord_dp, cache_key="d_sw-dp"
    )
    fvtp2d_vt = utils.cached_stencil_class(FiniteVolumeTransport)(
        spec.namelist, spec.namelist.hord_vt, cache_key="d_sw-vt"
    )
    fvtp2d_tm = utils.cached_stencil_class(FiniteVolumeTransport)(
        spec.namelist, spec.namelist.hord_tm, cache_key="d_sw-tm"
    )

    fxadv.compute(uc, vc, crx, cry, xfx, yfx, ut, vt, dt)

    fvtp2d_dp(
        delp,
        crx,
        cry,
        xfx,
        yfx,
        fx,
        fy,
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
    )

    flux_capacitor(
        cx,
        cy,
        mfx,
        mfy,
        crx,
        cry,
        fx,
        fy,
        origin=spec.grid.full_origin(),
        domain=spec.grid.domain_shape_full(),
    )

    if not spec.namelist.hydrostatic:
        dw, wk = damp_vertical_wind(w, heat_s, diss_est, dt, column_namelist)
        fvtp2d_vt(
            w,
            crx,
            cry,
            xfx,
            yfx,
            gx,
            gy,
            nord=column_namelist["nord_v"],
            damp_c=column_namelist["damp_vt"],
            mfx=fx,
            mfy=fy,
        )

        flux_adjust(
            w,
            delp,
            gx,
            gy,
            grid().rarea,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )
    # USE_COND
    fvtp2d_dp(
        q_con,
        crx,
        cry,
        xfx,
        yfx,
        gx,
        gy,
        nord=column_namelist["nord_t"],
        damp_c=column_namelist["damp_t"],
        mass=delp,
        mfx=fx,
        mfy=fy,
    )

    flux_adjust(
        q_con,
        delp,
        gx,
        gy,
        grid().rarea,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(),
    )

    # END USE_COND

    fvtp2d_tm(
        pt,
        crx,
        cry,
        xfx,
        yfx,
        gx,
        gy,
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
        mass=delp,
        mfx=fx,
        mfy=fy,
    )

    if spec.namelist.inline_q:
        raise Exception("inline_q not yet implemented")

    dt5 = 0.5 * dt
    dt4 = 0.25 * dt

    not_inlineq_pressure_and_vbke(
        gx,
        gy,
        grid().rarea,
        fx,
        fy,
        pt,
        delp,
        vc,
        uc,
        grid().cosa,
        grid().rsina,
        vt,
        vb,
        dt4,
        dt5,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )
    ytp_v_obj = utils.cached_stencil_class(YTP_V)(spec.namelist, cache_key="ytp_v")
    ytp_v_obj(vb, v, ub)

    mult_ubke(
        vb,
        ke,
        uc,
        vc,
        grid().cosa,
        grid().rsina,
        ut,
        ub,
        dt4,
        dt5,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )
    xtp_u_obj = utils.cached_stencil_class(XTP_U)(spec.namelist, cache_key="xtp_u")
    xtp_u_obj(ub, u, vb)

    ke_horizontal_vorticity(
        ke,
        u,
        v,
        ub,
        vb,
        ut,
        vt,
        spec.grid.dx,
        spec.grid.dy,
        spec.grid.rarea,
        wk,
        dt,
        origin=grid().full_origin(),
        domain=grid().domain_shape_full(),
    )

    # TODO if spec.namelist.d_f3d and ROT3 unimplemeneted
    adjust_w_and_qcon(
        w,
        delp,
        dw,
        q_con,
        column_namelist["damp_w"],
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(),
    )
    for kstart, nk in k_bounds():
        divdamp.compute(
            u,
            v,
            va,
            ptc,
            vort,
            ua,
            divgd,
            vc,
            uc,
            delpc,
            ke,
            wk,
            column_namelist["d2_divg"][kstart],
            dt,
            column_namelist["nord"][kstart],
            kstart=kstart,
            nk=nk,
        )

    ub_vb_from_vort(
        vort,
        ub,
        vb,
        column_namelist["d_con"],
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    # Vorticity transport
    compute_vorticity(
        wk,
        grid().f0,
        zh,
        vort,
        origin=grid().full_origin(),
        domain=grid().domain_shape_full(),
    )

    fvtp2d_vt(vort, crx, cry, xfx, yfx, fx, fy)

    u_and_v_from_ke(
        ke,
        ut,
        vt,
        fx,
        fy,
        u,
        v,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    for kstart, nk in k_bounds():
        if column_namelist["damp_vt"][kstart] > dcon_threshold:
            damp4 = (column_namelist["damp_vt"][kstart] * grid().da_min_c) ** (
                column_namelist["nord_v"][kstart] + 1
            )
            delnflux.compute_no_sg(
                wk,
                ut,
                vt,
                column_namelist["nord_v"][kstart],
                damp4,
                vort,
                kstart=kstart,
                nk=nk,
            )

    heat_source_from_vorticity_damping(
        ub,
        vb,
        ut,
        vt,
        u,
        v,
        delp,
        grid().rsin2,
        grid().cosa_s,
        grid().rdx,
        grid().rdy,
        heat_s,
        heat_source,
        diss_est,
        column_namelist["d_con"],
        column_namelist["damp_vt"],
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )
