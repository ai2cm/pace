from typing import Dict, List

import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.divergence_damping as divdamp
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.stencils.fxadv as fxadv
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.xtp_u import XTP_U
from fv3core.stencils.ytp_v import YTP_V
from fv3core.utils.typing import FloatField, FloatFieldIJ


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
        assert __INLINED(spec.namelist.grid_type < 3)
        vb = vbke(vc, uc, cosa, rsina, vt, vb, dt4, dt5)


@gtscript.function
def ke_from_bwind(ke, ub, vb):
    return 0.5 * (ke + ub * vb)


@gtstencil()
def ub_vb_from_vort(
    vort: FloatField,
    ub: FloatField,
    vb: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
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
@gtstencil(externals={"radius": constants.RADIUS})
def coriolis_force_correction(zh: FloatField, z_rat: FloatField):
    from __externals__ import radius

    with computation(PARALLEL), interval(...):
        z_rat = 1.0 + (zh + zh[0, 0, 1]) / radius


@gtstencil()
def zrat_vorticity(
    wk: FloatField,
    f0: FloatFieldIJ,
    z_rat: FloatField,
    vort: FloatField,
):

    from __externals__ import namelist

    with computation(PARALLEL), interval(...):
        if __INLINED(namelist.do_f3d and not namelist.hydrostatic):
            vort = wk + f0 * z_rat
        else:
            vort = wk[0, 0, 0] + f0[0, 0]


@gtscript.function
def add_dw(w, dw, damp_w):
    w = w + dw if damp_w > 1e-5 else w
    return w


@gtstencil()
def adjust_w_and_qcon(
    w: FloatField, delp: FloatField, dw: FloatField, q_con: FloatField, damp_w: float
):
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = add_dw(w, dw, damp_w)
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
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        dw = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
        heat_source = dd8 - dw * (w + 0.5 * dw)
        diss_est = heat_source


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
    dissipation_estimate: FloatField,
    kinetic_energy_fraction_to_damp: float,
):
    """
    Calculates heat source from vorticity damping implied by energy conservation.

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
        dissipation_estimate (out): dissipation estimate, only calculated if
            calculate_dissipation_estimate is 1
        kinetic_energy_fraction_to_damp (in): according to its comment in fv_arrays,
            the fraction of kinetic energy to explicitly damp and convert into heat.
            TODO: confirm this description is accurate, why is it multiplied
            by 0.25 below?
    """
    from __externals__ import namelist

    with computation(PARALLEL), interval(...):
        ubt = (ub + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vb - ut) * rdy
        fx = v * rdy
        gx = fx * vbt
        u2 = fy + fy[0, 1, 0]
        du2 = ubt + ubt[0, 1, 0]
        v2 = fx + fx[1, 0, 0]
        dv2 = vbt + vbt[1, 0, 0]
        dampterm = heat_damping_term(ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2)
        heat_source = delp * (
            heat_source - 0.25 * kinetic_energy_fraction_to_damp * dampterm
        )
        # do_skeb could be renamed to calculate_dissipation_estimate
        # when d_sw is converted into a D_SW object
        if __INLINED(namelist.do_skeb == 1):
            dissipation_estimate = -dampterm


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
    col = {}
    num_k = len(k_bounds())
    for name in direct_namelist:
        col[name] = [getattr(spec.namelist, name)] * num_k

    col["d2_divg"] = [min(0.2, spec.namelist.d2_bg)] * num_k
    col["nord_v"] = [min(2, col["nord"][i]) for i in range(num_k)]
    col["nord_w"] = [val for val in col["nord_v"]]
    col["nord_t"] = [val for val in col["nord_v"]]
    if spec.namelist.do_vort_damp:
        col["damp_vt"] = [spec.namelist.vtdm4] * num_k
    else:
        col["damp_vt"] = [0] * num_k
    col["damp_w"] = [val for val in col["damp_vt"]]
    col["damp_t"] = [val for val in col["damp_vt"]]
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

    # TODO: Remove paired with removal of #d_sw belos
    # column_namelist = column_namelist_options(0)
    column_namelist = get_column_namelist()
    heat_s = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    diss_e = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    z_rat = utils.make_storage_from_shape(heat_source.shape, grid().full_origin())
    # TODO: If namelist['hydrostatic' and not namelist['use_old_omega'] and last_step.
    if spec.namelist.d_ext > 0:
        raise Exception(
            "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        )
    if spec.namelist.do_f3d and not spec.namelist.hydrostatic:
        coriolis_force_correction(
            zh,
            z_rat,
            origin=grid().full_origin(),
            domain=grid().domain_shape_full(),
        )

    d_sw(
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
        z_rat,
        heat_s,
        diss_e,
        dt,
        column_namelist,
    )

    # TODO: If namelist['hydrostatic' and not namelist['use_old_omega'] and last_step.

    # TODO: If namelist['d_ext'] > 0

    if spec.namelist.d_con > dcon_threshold or spec.namelist.do_skeb:
        basic.add_term_two_vars(
            heat_s,
            heat_source,
            diss_e,
            diss_est,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )


def damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist, kstart, nk):
    dw = utils.make_storage_from_shape(w.shape, grid().compute_origin())
    wk = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fx2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fy2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    if column_namelist["damp_w"][kstart] > 1e-5:
        dd8 = column_namelist["ke_bg"][kstart] * abs(dt)
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
            diss_e,
            dw,
            dd8,
            origin=(grid().is_, grid().js, kstart),
            domain=(grid().nic, grid().njc, nk),
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
        assert __INLINED(spec.namelist.grid_type < 3)
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


def d_sw(
    delpc: FloatField,
    delp: FloatField,
    ptc: FloatField,
    pt: FloatField,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    uc: FloatField,
    vc: FloatField,
    ua: FloatField,
    va: FloatField,
    divgd: FloatField,
    xflux: FloatField,
    yflux: FloatField,
    cx: FloatField,
    cy: FloatField,
    crx: FloatField,
    cry: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    q_con: FloatField,
    z_rat: FloatField,
    heat_s: FloatField,
    diss_e: FloatField,
    dt: float,
    column_namelist: Dict[str, List],
):
    shape = heat_s.shape
    ub = utils.make_storage_from_shape(shape, grid().compute_origin())
    vb = utils.make_storage_from_shape(shape, grid().compute_origin())
    ke = utils.make_storage_from_shape(shape, grid().full_origin())
    vort = utils.make_storage_from_shape(shape, grid().full_origin())
    ut = utils.make_storage_from_shape(shape, grid().full_origin())
    vt = utils.make_storage_from_shape(shape, grid().full_origin())
    fx = utils.make_storage_from_shape(shape, grid().compute_origin())
    fy = utils.make_storage_from_shape(shape, grid().compute_origin())
    gx = utils.make_storage_from_shape(shape, grid().compute_origin())
    gy = utils.make_storage_from_shape(shape, grid().compute_origin())
    fvtp2d_dp = utils.cached_stencil_class(fvtp2d.FvTp2d)(
        spec.namelist, spec.namelist.hord_dp, cache_key="d_sw-dp"
    )
    fvtp2d_vt = utils.cached_stencil_class(fvtp2d.FvTp2d)(
        spec.namelist, spec.namelist.hord_vt, cache_key="d_sw-vt"
    )
    fvtp2d_tm = utils.cached_stencil_class(fvtp2d.FvTp2d)(
        spec.namelist, spec.namelist.hord_tm, cache_key="d_sw-tm"
    )
    ra_x, ra_y = fxadv.compute(uc, vc, ut, vt, xfx, yfx, crx, cry, dt)

    fvtp2d_dp(
        delp,
        crx,
        cry,
        xfx,
        yfx,
        ra_x,
        ra_y,
        fx,
        fy,
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
    )

    flux_capacitor(
        cx,
        cy,
        xflux,
        yflux,
        crx,
        cry,
        fx,
        fy,
        origin=spec.grid.full_origin(),
        domain=spec.grid.domain_shape_full(),
    )

    if not spec.namelist.hydrostatic:
        for kstart, nk in k_bounds():
            dw, wk = damp_vertical_wind(
                w, heat_s, diss_e, dt, column_namelist, kstart, nk
            )

        fvtp2d_vt(
            w,
            crx,
            cry,
            xfx,
            yfx,
            ra_x,
            ra_y,
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
        ra_x,
        ra_y,
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
        ra_x,
        ra_y,
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
        origin=(0, 0, 0),
        domain=spec.grid.domain_shape_full(),
    )

    # TODO if spec.namelist.d_f3d and ROT3 unimplemeneted
    for kstart, nk in k_bounds():
        adjust_w_and_qcon(
            w,
            delp,
            dw,
            q_con,
            column_namelist["damp_w"][kstart],
            origin=(grid().is_, grid().js, kstart),
            domain=(grid().nic, grid().njc, nk),
        )

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

        if column_namelist["d_con"][kstart] > dcon_threshold:
            ub_vb_from_vort(
                vort,
                ub,
                vb,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic + 1, grid().njc + 1, nk),
            )

    # Vorticity transport
    zrat_vorticity(
        wk,
        grid().f0,
        z_rat,
        vort,
        origin=grid().full_origin(),
        domain=grid().domain_shape_full(),
    )

    fvtp2d_vt(vort, crx, cry, xfx, yfx, ra_x, ra_y, fx, fy)

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

            if (
                column_namelist["d_con"][kstart] > dcon_threshold
                or spec.namelist.do_skeb
            ):
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
                    diss_e,
                    column_namelist["d_con"][kstart],
                    origin=(grid().is_, grid().js, kstart),
                    domain=(grid().nic, grid().njc, nk),
                )

        if column_namelist["damp_vt"][kstart] > 1e-5:
            basic.add_term_stencil(
                vt,
                u,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic, grid().njc + 1, nk),
            )
            basic.subtract_term_stencil(
                ut,
                v,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic + 1, grid().njc, nk),
            )
