#!/usr/bin/env python3
import copy
import math
from types import SimpleNamespace

import fv3gfs.util as fv3util
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.c_sw as c_sw
import fv3core.stencils.copy_stencil as cp
import fv3core.stencils.d2a2c_vect as d2a2c
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.del2cubed as del2cubed
import fv3core.stencils.nh_p_grad as nh_p_grad
import fv3core.stencils.pe_halo as pe_halo
import fv3core.stencils.pgradc as pgradc
import fv3core.stencils.pk3_halo as pk3_halo
import fv3core.stencils.ray_fast as ray_fast
import fv3core.stencils.riem_solver3 as riem_solver3
import fv3core.stencils.riem_solver_c as riem_solver_c
import fv3core.stencils.temperature_adjust as temperature_adjust
import fv3core.stencils.updatedzc as updatedzc
import fv3core.stencils.updatedzd as updatedzd
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils


sd = utils.sd
HUGE_R = 1.0e40

# NOTE in Fortran these are columns
@utils.stencil()
def dp_ref_compute(ak: sd, bk: sd, dp_ref: sd):
    with computation(PARALLEL), interval(0, -1):
        dp_ref = ak[0, 0, 1] - ak + (bk[0, 0, 1] - bk) * 1.0e5


@utils.stencil()
def set_gz(zs: sd, delz: sd, gz: sd):
    with computation(BACKWARD):
        with interval(-1, None):
            gz[0, 0, 0] = zs
        with interval(0, -1):
            gz[0, 0, 0] = gz[0, 0, 1] - delz


@utils.stencil()
def set_pem(delp: sd, pem: sd, ptop: float):
    with computation(FORWARD):
        with interval(0, 1):
            pem[0, 0, 0] = ptop
        with interval(1, None):
            pem[0, 0, 0] = pem[0, 0, -1] + delp


@utils.stencil()
def heatadjust_temperature_lowlevel(
    pt: sd, heat_source: sd, delp: sd, pkz: sd, cp_air: float
):
    with computation(PARALLEL), interval(...):
        pt[0, 0, 0] = pt + heat_source / (cp_air * delp * pkz)


def get_n_con():
    if spec.namelist.convert_ke or spec.namelist.vtdm4 > 1.0e-4:
        n_con = spec.grid.npz
    else:
        if spec.namelist.d2_bg_k1 < 1.0e-3:
            n_con = 0
        else:
            if spec.namelist.d2_bg_k2 < 1.0e-3:
                n_con = 1
            else:
                n_con = 2
    return n_con


def dyncore_temporaries(shape):
    grid = spec.grid
    tmps = {}
    utils.storage_dict(
        tmps,
        ["ut", "vt", "gz", "zh", "pem", "ws3", "pkc", "pk3", "heat_source", "divgd"],
        shape,
        grid.default_origin(),
    )
    utils.storage_dict(tmps, ["crx", "xfx"], shape, grid.compute_x_origin())
    utils.storage_dict(tmps, ["cry", "yfx"], shape, grid.compute_y_origin())
    grid.quantity_dict_update(
        tmps, "heat_source", dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM]
    )
    for q in ["gz", "pkc", "zh"]:
        grid.quantity_dict_update(
            tmps, q, dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM]
        )
    grid.quantity_dict_update(
        tmps,
        "divgd",
        dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
    )
    return tmps


def compute(state, comm):
    # u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd, mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split, akap, ptop, pfull, n_map, comm):
    grid = spec.grid

    init_step = state.n_map == 1
    end_step = state.n_map == spec.namelist.k_split
    akap = state.akap
    # peln1 = math.log(ptop)
    # ptk = ptop**akap
    dt = state.mdt / state.n_split
    dt2 = 0.5 * dt
    hydrostatic = spec.namelist.hydrostatic
    rgrav = 1.0 / constants.GRAV
    n_split = state.n_split
    # TODO -- put defaults into code
    # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
    # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
    ms = max(1, spec.namelist.m_split / 2.0)
    shape = state.delz.shape
    # NOTE in Fortran model the halo update starts happens in fv_dynamics, not here
    reqs = {}
    for halovar in ["q_con_quantity", "cappa_quantity", "delp_quantity", "pt_quantity"]:
        reqs[halovar] = comm.start_halo_update(
            state.__getattribute__(halovar), n_points=utils.halo
        )
    reqs_vector = comm.start_vector_halo_update(
        state.u_quantity, state.v_quantity, n_points=utils.halo
    )
    reqs["q_con_quantity"].wait()
    reqs["cappa_quantity"].wait()

    state.__dict__.update(dyncore_temporaries(shape))
    if init_step:
        state.gz[:-1, :-1, :] = HUGE_R
        state.diss_estd[grid.slice_dict(grid.compute_dict())] = 0.0
        if not hydrostatic:
            state.pk3[:-1, :-1, :] = HUGE_R
    state.mfxd[grid.slice_dict(grid.x3d_compute_dict())] = 0.0
    state.mfyd[grid.slice_dict(grid.y3d_compute_dict())] = 0.0
    state.cxd[grid.slice_dict(grid.x3d_compute_domain_y_dict())] = 0.0
    state.cyd[grid.slice_dict(grid.y3d_compute_domain_x_dict())] = 0.0
    if not hydrostatic:
        # k1k = akap / (1.0 - akap)
        # TODO -- is really just a column... when different shapes are supported perhaps change this
        state.dp_ref = utils.make_storage_from_shape(
            state.ak.shape, grid.default_origin()
        )
        dp_ref_compute(state.ak, state.bk, state.dp_ref)
        state.zs = state.phis * rgrav
    n_con = get_n_con()

    for it in range(n_split):
        remap_step = False
        if spec.namelist.breed_vortex_inline or (it == n_split - 1):
            remap_step = True
        if not hydrostatic:
            reqs["w_quantity"] = comm.start_halo_update(
                state.w_quantity, n_points=utils.halo
            )
            if it == 0:
                set_gz(
                    state.zs,
                    state.delz,
                    state.gz,
                    origin=grid.compute_origin(),
                    domain=(grid.nic, grid.njc, grid.npz + 1),
                )
                reqs["gz_quantity"] = comm.start_halo_update(
                    state.gz_quantity, n_points=utils.halo
                )
        if it == 0:
            reqs["delp_quantity"].wait()
            reqs["pt_quantity"].wait()
            beta_d = 0
        else:
            beta_d = spec.namelist.beta
        last_step = False
        if it == n_split - 1 and end_step:
            last_step = True

        if it == n_split - 1 and end_step:
            if spec.namelist.use_old_omega:  # apparently True
                set_pem(
                    state.delp,
                    state.pem,
                    state.ptop,
                    origin=(grid.is_ - 1, grid.js - 1, 0),
                    domain=(grid.nic + 2, grid.njc + 2, grid.npz),
                )
        reqs_vector.wait()
        if not hydrostatic:
            reqs["w_quantity"].wait()

        state.delpc, state.ptc = c_sw.compute(
            state.delp,
            state.pt,
            state.u,
            state.v,
            state.w,
            state.uc,
            state.vc,
            state.ua,
            state.va,
            state.ut,
            state.vt,
            state.divgd,
            state.omga,
            dt2,
        )

        if spec.namelist.nord > 0:
            reqs["divgd_quantity"] = comm.start_halo_update(
                state.divgd_quantity, n_points=utils.halo
            )
        if not hydrostatic:
            if it == 0:
                reqs["gz_quantity"].wait()
                cp.copy_stencil(
                    state.gz,
                    state.zh,
                    origin=grid.default_origin(),
                    domain=grid.domain_shape_buffer_k(),
                )
            else:
                cp.copy_stencil(
                    state.gz,
                    state.zh,
                    origin=grid.default_origin(),
                    domain=grid.domain_shape_buffer_k(),
                )
        if not hydrostatic:
            state.gz, state.ws3 = updatedzc.compute(
                state.dp_ref, state.zs, state.ut, state.vt, state.gz, state.ws3, dt2
            )
            # TODO this is really a 2d field.
            state.ws3 = utils.make_storage_data_from_2d(
                state.ws3[:, :, -1], shape, origin=(0, 0, 0)
            )
            riem_solver_c.compute(
                ms,
                dt2,
                akap,
                state.cappa,
                state.ptop,
                state.phis,
                state.omga,
                state.ptc,
                state.q_con,
                state.delpc,
                state.gz,
                state.pkc,
                state.ws3,
            )

        pgradc.compute(state.uc, state.vc, state.delpc, state.pkc, state.gz, dt2)
        reqc_vector = comm.start_vector_halo_update(
            state.uc_quantity, state.vc_quantity, n_points=utils.halo
        )
        if spec.namelist.nord > 0:
            reqs["divgd_quantity"].wait()
        reqc_vector.wait()
        state.nord_v, state.damp_vt = d_sw.compute(
            state.vt,
            state.delp,
            state.ptc,
            state.pt,
            state.u,
            state.v,
            state.w,
            state.uc,
            state.vc,
            state.ua,
            state.va,
            state.divgd,
            state.mfxd,
            state.mfyd,
            state.cxd,
            state.cyd,
            state.crx,
            state.cry,
            state.xfx,
            state.yfx,
            state.q_con,
            state.zh,
            state.heat_source,
            state.diss_estd,
            dt,
        )

        for halovar in ["delp_quantity", "pt_quantity", "q_con_quantity"]:
            comm.halo_update(state.__getattribute__(halovar), n_points=utils.halo)

        # Not used unless we implement other betas and alternatives to nh_p_grad
        # if spec.namelist.d_ext > 0:
        #    raise 'Unimplemented namelist option d_ext > 0'
        # else:
        #    divg2 = utils.make_storage_from_shape(delz.shape, grid.compute_origin())

        if not hydrostatic:
            updatedzd.compute(
                state.nord_v,
                state.damp_vt,
                state.dp_ref,
                state.zs,
                state.zh,
                state.crx,
                state.cry,
                state.xfx,
                state.yfx,
                state.wsd,
                dt,
            )

            # TODO this is really a 2d field.
            state.wsd = utils.make_storage_data_from_2d(
                state.wsd[:, :, -1], shape, origin=grid.compute_origin()
            )
            riem_solver3.compute(
                remap_step,
                dt,
                akap,
                state.cappa,
                state.ptop,
                state.zs,
                state.w,
                state.delz,
                state.q_con,
                state.delp,
                state.pt,
                state.zh,
                state.pe,
                state.pkc,
                state.pk3,
                state.pk,
                state.peln,
                state.wsd,
            )

            reqs["zh_quantity"] = comm.start_halo_update(
                state.zh_quantity, n_points=utils.halo
            )
            if grid.npx == grid.npy:
                reqs["pkc_quantity"] = comm.start_halo_update(
                    state.pkc_quantity, n_points=2
                )
            else:
                reqs["pkc_quantity"] = comm.start_halo_update(
                    state.pkc_quantity, n_points=utils.halo
                )
            if remap_step:
                pe_halo.compute(state.pe, state.delp, state.ptop)
            if spec.namelist.use_logp:
                raise Exception("unimplemented namelist option use_logp=True")
            else:
                pk3_halo.compute(state.pk3, state.delp, state.ptop, akap)
        if not hydrostatic:
            reqs["zh_quantity"].wait()
            if grid.npx != grid.npy:
                reqs["pkc_quantity"].wait()
        if not hydrostatic:
            basic.multiply_constant(
                state.zh,
                constants.GRAV,
                state.gz,
                origin=(grid.is_ - 2, grid.js - 2, 0),
                domain=(grid.nic + 4, grid.njc + 4, grid.npz + 1),
            )
            if grid.npx == grid.npy:
                reqs["pkc_quantity"].wait()
            if spec.namelist.beta != 0:
                raise Exception(
                    "Unimplemented namelist option -- we only support beta=0"
                )
        if not hydrostatic:
            nh_p_grad.compute(
                state.u,
                state.v,
                state.pkc,
                state.gz,
                state.pk3,
                state.delp,
                dt,
                state.ptop,
                akap,
            )

        if spec.namelist.rf_fast:
            # TODO pass through ks, or remove, inconsistent representation vs Fortran
            ray_fast.compute(
                state.u,
                state.v,
                state.w,
                state.dp_ref,
                state.pfull,
                dt,
                state.ptop,
                state.ks,
            )

        if it != n_split - 1:
            reqs_vector = comm.start_vector_halo_update(
                state.u_quantity, state.v_quantity, n_points=utils.halo
            )
        else:
            if spec.namelist.grid_type < 4:
                comm.synchronize_vector_interfaces(state.u_quantity, state.v_quantity)

    if n_con != 0 and spec.namelist.d_con > 1.0e-5:
        nf_ke = min(3, spec.namelist.nord + 1)

        comm.halo_update(state.heat_source_quantity, n_points=utils.halo)
        cd = constants.CNST_0P20 * grid.da_min
        del2cubed.compute(state.heat_source, nf_ke, cd, grid.npz)
        if not hydrostatic:
            temperature_adjust.compute(
                state.pt,
                state.pkz,
                state.heat_source,
                state.delz,
                state.delp,
                state.cappa,
                n_con,
                dt,
            )
