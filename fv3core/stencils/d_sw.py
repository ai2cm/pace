#!/usr/bin/env python3
import logging

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.divergence_damping as divdamp
import fv3core.stencils.flux_capacitor as fluxcap
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.stencils.fxadv as fxadv
import fv3core.stencils.heatdiss as heatdiss
import fv3core.stencils.ubke as ubke
import fv3core.stencils.vbke as vbke
import fv3core.stencils.vorticity_volumemean as vort_mean
import fv3core.stencils.xtp_u as xtp_u
import fv3core.stencils.ytp_v as ytp_v
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


dcon_threshold = 1e-5
sd = utils.sd

logger = logging.getLogger("fv3ser")


def grid():
    return spec.grid


def k_indices():
    return [[0, 1], [1, 2], [2, 3], list(range(3, grid().npz + 1))]


def d_sw_ksplit(func, data, splitvars_values, outputs, grid, allz=False):
    utils.k_split_run_dataslice(
        func, data, k_indices(), splitvars_values, outputs, grid, allz
    )


@gtscript.function
def flux_component(gx, gy, rarea):
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


@gtscript.function
def flux_integral(w, delp, gx, gy, rarea):
    return w * delp + flux_component(gx, gy, rarea)


@gtstencil()
def flux_adjust(w: sd, delp: sd, gx: sd, gy: sd, rarea: sd):
    with computation(PARALLEL), interval(...):
        w = flux_integral(w, delp, gx, gy, rarea)


@gtstencil()
def not_inlineq_pressure(gx: sd, gy: sd, rarea: sd, fx: sd, fy: sd, pt: sd, delp: sd):
    with computation(PARALLEL), interval(...):
        pt = flux_integral(
            pt, delp, gx, gy, rarea
        )  # TODO: put [0, 0, 0] on left when gt4py bug is fixed
        delp = delp + flux_component(
            fx, fy, rarea
        )  # TODO: put [0, 0, 0] on left when gt4py bug is fixed
        pt[0, 0, 0] = pt / delp


@gtstencil()
def ke_from_bwind(ke: sd, ub: sd, vb: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = 0.5 * (ke + ub * vb)


@gtstencil()
def ub_from_vort(vort: sd, ub: sd):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = vort - vort[1, 0, 0]


@gtstencil()
def vb_from_vort(vort: sd, vb: sd):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = vort - vort[0, 1, 0]


@gtstencil()
def u_from_ke(ke: sd, vt: sd, fy: sd, u: sd):
    with computation(PARALLEL), interval(...):
        u[0, 0, 0] = vt + ke - ke[1, 0, 0] + fy


@gtstencil()
def v_from_ke(ke: sd, ut: sd, fx: sd, v: sd):
    with computation(PARALLEL), interval(...):
        v[0, 0, 0] = ut + ke - ke[0, 1, 0] - fx


# TODO: this is untested and the radius may be incorrect
@gtstencil(externals={"radius": constants.RADIUS})
def coriolis_force_correction(zh: sd, z_rat: sd):
    from __externals__ import radius

    with computation(PARALLEL), interval(...):
        z_rat[0, 0, 0] = 1.0 + (zh + zh[0, 0, 1]) / radius


@gtstencil()
def zrat_vorticity(wk: sd, f0: sd, z_rat: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = wk + f0 * z_rat


@gtscript.function
def add_dw(w, dw, damp_w):
    w = w + dw if damp_w > 1e-5 else w
    return w


@gtstencil()
def adjust_w_and_qcon(w: sd, delp: sd, dw: sd, q_con: sd, damp_w: float):
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = add_dw(w, dw, damp_w)
        # USE_COND
        q_con = q_con / delp


@gtstencil()
def heatdamping_setup(ub: sd, vt: sd, fy: sd, u: sd, gy: sd, rdx: sd, sign: float):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = (ub + sign * vt) * rdx
        fy[0, 0, 0] = u * rdx
        gy[0, 0, 0] = fy * ub


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


@gtstencil()
def heatdamping(
    ub: sd,
    vb: sd,
    delp: sd,
    fx: sd,
    fy: sd,
    gx: sd,
    gy: sd,
    rsin2: sd,
    cosa_s: sd,
    heat_source: sd,
    diss_est: sd,
    damp: float,
    do_skeb: int,
):
    with computation(PARALLEL), interval(...):
        u2 = fy + fy[0, 1, 0]
        du2 = ub + ub[0, 1, 0]
        v2 = fx + fx[1, 0, 0]
        dv2 = vb + vb[1, 0, 0]
        dampterm = heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2)
        heat_source[0, 0, 0] = delp * (heat_source - damp * dampterm)
        diss_est[0, 0, 0] = diss_est - dampterm if do_skeb == 1 else diss_est


def initialize_heat_source(heat_source, diss_est):
    heat_source[grid().compute_interface()] = 0
    diss_est[grid().compute_interface()] = 0


def heat_from_damping(
    ub, vb, ut, vt, u, v, delp, fx, fy, gx, gy, heat_source, diss_est, damp
):
    heatdamping_setup(
        ub,
        vt,
        fy,
        u,
        gy,
        grid().rdx,
        1.0,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_y(),
    )
    heatdamping_setup(
        vb,
        ut,
        fx,
        v,
        gx,
        grid().rdy,
        -1.0,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_x(),
    )
    heatdamping(
        ub,
        vb,
        delp,
        fx,
        fy,
        gx,
        gy,
        grid().rsin2,
        grid().cosa_s,
        heat_source,
        diss_est,
        damp,
        int(spec.namelist.do_skeb),
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(),
    )


def set_low_kvals(col):
    for name in ["nord", "nord_w", "d_con"]:
        col[name] = 0
    col["damp_w"] = col["d2_divg"]


def vort_damp_option(col):
    if spec.namelist.do_vort_damp:
        col["nord_v"] = 0
        col["damp_vt"] = 0.5 * col["d2_divg"]


def lowest_kvals(col):
    set_low_kvals(col)
    vort_damp_option(col)


def max_d2_bg0():
    return max(0.01, spec.namelist.d2_bg, spec.namelist.d2_bg_k1)


def max_d2_bg1():
    return max(spec.namelist.d2_bg, spec.namelist.d2_bg_k2)


def get_column_namelist():
    ks = [k[0] for k in k_indices()]
    col = {"column_namelist": []}
    for ki in ks:
        col["column_namelist"].append(column_namelist_options(ki))
    return col


def get_single_column(key):
    col = []
    for k in range(0, grid().npz):
        col.append(column_namelist_options(k)[key])
    col.append(0.0)
    return col


def column_namelist_options(k):
    direct_namelist = ["ke_bg", "d_con", "nord"]
    col = {}
    for name in direct_namelist:
        col[name] = getattr(spec.namelist, name)
    col["d2_divg"] = min(0.2, spec.namelist.d2_bg)
    col["nord_v"] = min(2, col["nord"])
    col["nord_w"] = col["nord_v"]
    col["nord_t"] = col["nord_v"]
    if spec.namelist.do_vort_damp:
        col["damp_vt"] = spec.namelist.vtdm4
    else:
        col["damp_vt"] = 0
    col["damp_w"] = col["damp_vt"]
    col["damp_t"] = col["damp_vt"]
    if grid().npz == 1 or spec.namelist.n_sponge < 0:
        pass
    #     d2_divg = spec.namelist.d2_bg  # commenting because unused, never gets set into col
    else:
        if k == 0:
            col["d2_divg"] = max_d2_bg0()
            lowest_kvals(col)
        if k == 1 and spec.namelist.d2_bg_k2 > 0.01:
            col["d2_divg"] = max_d2_bg1()
            lowest_kvals(col)
        if k == 2 and spec.namelist.d2_bg_k2 > 0.05:
            col["d2_divg"] = max(spec.namelist.d2_bg, 0.2 * spec.namelist.d2_bg_k2)
            set_low_kvals(col)
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

    # TODO: remove paired with removal of #d_sw belos
    # column_namelist = column_namelist_options(0)
    column_namelist = get_column_namelist()
    heat_s = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    diss_e = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    z_rat = utils.make_storage_from_shape(heat_source.shape, grid().default_origin())
    # TODO if namelist['hydrostatic' and not namelist['use_old_omega'] and last_step
    if spec.namelist.d_ext > 0:
        raise Exception(
            "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        )
    if spec.namelist.do_f3d and not spec.namelist.hydrostatic:
        coriolis_force_correction(
            zh,
            z_rat,
            origin=grid().default_origin(),
            domain=grid().domain_shape_standard(),
        )
    # TODO: this seems a little redundant, revisit the k column split mechanism and/or the argument passing method
    in_only_vars = ["z_rat", "dt"]
    xflux = mfx
    yflux = mfy
    inout_vars = [
        "delpc",
        "delp",
        "ptc",
        "pt",
        "u",
        "v",
        "w",
        "uc",
        "vc",
        "ua",
        "va",
        "divgd",
        "xflux",
        "yflux",
        "cx",
        "cy",
        "crx",
        "cry",
        "xfx",
        "yfx",
        "q_con",
        "heat_s",
        "diss_e",
    ]
    data = {}
    for varname in inout_vars + in_only_vars:
        data[varname] = locals()[varname]
    outputs = {}
    for iv in inout_vars:
        outputs[iv] = data[iv]
    d_sw_ksplit(d_sw, data, column_namelist, outputs, grid())
    # TODO: remove when it has been decided how to handle the parameter arguments that change in the vertical. helpful for debugging
    # d_sw(delpc, delp, ptc, pt, u, v, w, uc, vc,  ua, va, divgd, mfx, mfy, cx, cy,  crx, cry, xfx, yfx, q_con, z_rat, heat_s, diss_e, dt,column_namelist)
    # TODO if namelist['hydrostatic' and not namelist['use_old_omega'] and last_step
    # TODO if namelist['d_ext'] > 0

    if spec.namelist.d_con > dcon_threshold or spec.namelist.do_skeb:
        basic.add_term_two_vars(
            heat_s,
            heat_source,
            diss_e,
            diss_est,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )
    nord_v = get_single_column("nord_v")
    damp_vt = get_single_column("damp_vt")
    return nord_v, damp_vt


def damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist):
    dw = utils.make_storage_from_shape(w.shape, grid().compute_origin())
    wk = utils.make_storage_from_shape(w.shape, grid().default_origin())
    fx2 = utils.make_storage_from_shape(w.shape, grid().default_origin())
    fy2 = utils.make_storage_from_shape(w.shape, grid().default_origin())
    if column_namelist["damp_w"] > 1e-5:
        dd8 = column_namelist["ke_bg"] * abs(dt)
        damp4 = (column_namelist["damp_w"] * grid().da_min_c) ** (
            column_namelist["nord_w"] + 1
        )

        delnflux.compute_no_sg(w, fx2, fy2, column_namelist["nord_w"], damp4, wk)
        heatdiss.compute(fx2, fy2, w, dd8, dw, heat_s, diss_e)
    return dw, wk


def d_sw(
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
    xflux,
    yflux,
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
):

    logger.debug("Parameters that vary with k: {}".format(column_namelist))
    shape = heat_s.shape
    ub = utils.make_storage_from_shape(shape, grid().compute_origin())
    vb = utils.make_storage_from_shape(shape, grid().compute_origin())
    ke = utils.make_storage_from_shape(shape, grid().default_origin())
    vort = utils.make_storage_from_shape(shape, grid().default_origin())
    ut = utils.make_storage_from_shape(shape, grid().default_origin())
    vt = utils.make_storage_from_shape(shape, grid().default_origin())
    fx = utils.make_storage_from_shape(shape, grid().compute_origin())
    fy = utils.make_storage_from_shape(shape, grid().compute_origin())
    gx = utils.make_storage_from_shape(shape, grid().compute_origin())
    gy = utils.make_storage_from_shape(shape, grid().compute_origin())
    ra_x, ra_y = fxadv.compute(uc, vc, ut, vt, xfx, yfx, crx, cry, dt)
    fvtp2d.compute_no_sg(
        delp,
        crx,
        cry,
        spec.namelist.hord_dp,
        xfx,
        yfx,
        ra_x,
        ra_y,
        fx,
        fy,
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
    )

    fluxcap.compute(cx, cy, xflux, yflux, crx, cry, fx, fy)
    initialize_heat_source(heat_s, diss_e)

    if not spec.namelist.hydrostatic:
        dw, wk = damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist)
        fvtp2d.compute_no_sg(
            w,
            crx,
            cry,
            spec.namelist.hord_vt,
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
    fvtp2d.compute_no_sg(
        q_con,
        crx,
        cry,
        spec.namelist.hord_dp,
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
    fvtp2d.compute_no_sg(
        pt,
        crx,
        cry,
        spec.namelist.hord_tm,
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
    else:
        not_inlineq_pressure(
            gx,
            gy,
            grid().rarea,
            fx,
            fy,
            pt,
            delp,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )

    dt5 = 0.5 * dt
    dt4 = 0.25 * dt
    vbke.compute(uc, vc, vt, vb, dt5, dt4)

    ytp_v.compute(vb, u, v, ub)

    basic.multiply_stencil(
        vb,
        ub,
        ke,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_buffer_2d(),
    )

    ubke.compute(uc, vc, ut, ub, dt5, dt4)

    xtp_u.compute(ub, u, v, vb)

    ke_from_bwind(
        ke,
        ub,
        vb,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_buffer_2d(),
    )

    if not grid().nested:
        corners.fix_corner_ke(ke, u, v, ut, vt, dt, grid())

    vort_mean.compute(u, v, ut, vt, wk)

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
        column_namelist["d2_divg"],
        dt,
        column_namelist["nord"],
    )

    if column_namelist["d_con"] > dcon_threshold:
        ub_from_vort(
            vort,
            ub,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute_y(),
        )
        vb_from_vort(
            vort,
            vb,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute_x(),
        )

    # Vorticity transport
    if spec.namelist.do_f3d and not spec.namelist.hydrostatic:
        zrat_vorticity(
            wk,
            grid().f0,
            z_rat,
            vort,
            orgin=grid().default_origin(),
            domain=grid().domain_shape_standard(),
        )
    else:
        basic.addition_stencil(
            wk,
            grid().f0,
            vort,
            origin=grid().default_origin(),
            domain=grid().domain_shape_standard(),
        )

    fvtp2d.compute_no_sg(
        vort, crx, cry, spec.namelist.hord_vt, xfx, yfx, ra_x, ra_y, fx, fy
    )

    u_from_ke(
        ke,
        vt,
        fy,
        u,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_y(),
    )

    v_from_ke(
        ke,
        ut,
        fx,
        v,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute_x(),
    )

    if column_namelist["damp_vt"] > dcon_threshold:
        damp4 = (column_namelist["damp_vt"] * grid().da_min_c) ** (
            column_namelist["nord_v"] + 1
        )
        delnflux.compute_no_sg(wk, ut, vt, column_namelist["nord_v"], damp4, vort)

    if column_namelist["d_con"] > dcon_threshold or spec.namelist.do_skeb:
        damp = 0.25 * column_namelist["d_con"]
        heat_from_damping(
            ub, vb, ut, vt, u, v, delp, fx, fy, gx, gy, heat_s, diss_e, damp
        )
    if column_namelist["damp_vt"] > 1e-5:
        basic.add_term_stencil(
            vt,
            u,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute_y(),
        )
        basic.subtract_term_stencil(
            ut,
            v,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute_x(),
        )
