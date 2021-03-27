from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.c_sw as c_sw
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.del2cubed as del2cubed
import fv3core.stencils.nh_p_grad as nh_p_grad
import fv3core.stencils.pe_halo as pe_halo
import fv3core.stencils.pk3_halo as pk3_halo
import fv3core.stencils.ray_fast as ray_fast
import fv3core.stencils.riem_solver3 as riem_solver3
import fv3core.stencils.riem_solver_c as riem_solver_c
import fv3core.stencils.temperature_adjust as temperature_adjust
import fv3core.stencils.updatedzc as updatedzc
import fv3core.stencils.updatedzd as updatedzd
import fv3core.utils.global_config as global_config
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util as fv3util
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


HUGE_R = 1.0e40


# NOTE in Fortran these are columns
@gtstencil()
def dp_ref_compute(
    ak: FloatFieldK,
    bk: FloatFieldK,
    phis: FloatFieldIJ,
    dp_ref: FloatField,
    zs: FloatField,
    rgrav: float,
):
    with computation(PARALLEL), interval(0, -1):
        dp_ref = ak[1] - ak + (bk[1] - bk) * 1.0e5
    with computation(PARALLEL), interval(...):
        zs = phis * rgrav


@gtstencil()
def set_gz(zs: FloatFieldIJ, delz: FloatField, gz: FloatField):
    with computation(BACKWARD):
        with interval(-1, None):
            gz[0, 0, 0] = zs
        with interval(0, -1):
            gz[0, 0, 0] = gz[0, 0, 1] - delz


@gtstencil()
def set_pem(delp: FloatField, pem: FloatField, ptop: float):
    with computation(FORWARD):
        with interval(0, 1):
            pem[0, 0, 0] = ptop
        with interval(1, None):
            pem[0, 0, 0] = pem[0, 0, -1] + delp


@gtstencil()
def heatadjust_temperature_lowlevel(
    pt: FloatField,
    heat_source: FloatField,
    delp: FloatField,
    pkz: FloatField,
    cp_air: float,
):
    with computation(PARALLEL), interval(...):
        pt[0, 0, 0] = pt + heat_source / (cp_air * delp * pkz)


@gtstencil()
def p_grad_c_stencil(
    rdxc: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    dt2: float,
):
    """Update C-grid winds from the pressure gradient force

    When this is run the C-grid winds have almost been completely
    updated by computing the momentum equation terms, but the pressure
    gradient force term has not yet been applied. This stencil completes
    the equation and Arakawa C-grid winds have been advected half a timestep
    upon completing this stencil..

     Args:
         uc: x-velocity on the C-grid (inout)
         vc: y-velocity on the C-grid (inout)
         delpc: vertical delta in pressure (in)
         pkc:  pressure if non-hydrostatic,
               (edge pressure)**(moist kappa) if hydrostatic(in)
         gz:  height of the model grid cells (m)(in)
         dt2: half a model timestep (for C-grid update) in seconds (in)
    Grid variable inputs:
        rdxc, rdyc
    """
    from __externals__ import local_ie, local_is, local_je, local_js, namelist

    with computation(PARALLEL), interval(...):
        if __INLINED(namelist.hydrostatic):
            wk = pkc[0, 0, 1] - pkc
        else:
            wk = delpc
        # TODO for PGradC validation only, not necessary for DynCore
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            uc = uc + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
                (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
                + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
            )
        # TODO for PGradC validation only, not necessary for DynCore
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            vc = vc + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
                (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
                + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
            )


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
        ["ut", "vt", "gz", "zh", "pem", "pkc", "pk3", "heat_source", "divgd"],
        shape,
        grid.full_origin(),
    )
    utils.storage_dict(
        tmps,
        ["ws3"],
        shape[0:2],
        grid.full_origin()[0:2],
    )
    utils.storage_dict(
        tmps, ["crx", "xfx"], shape, grid.compute_origin(add=(0, -grid.halo, 0))
    )
    utils.storage_dict(
        tmps, ["cry", "yfx"], shape, grid.compute_origin(add=(-grid.halo, 0, 0))
    )
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
    # u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd,
    # mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split,
    # akap, ptop, pfull, n_map, comm):
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
    # TODO: Put defaults into code.
    # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
    # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
    ms = max(1, spec.namelist.m_split / 2.0)
    shape = state.delz.shape
    # NOTE: In Fortran model the halo update starts happens in fv_dynamics, not here.
    reqs = {}
    if global_config.get_do_halo_exchange():
        for halovar in [
            "q_con_quantity",
            "cappa_quantity",
            "delp_quantity",
            "pt_quantity",
        ]:
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

        # To write in parallel region, these need to be 3D first
        state.dp_ref = utils.make_storage_from_shape(shape, grid.full_origin())
        state.zs = utils.make_storage_from_shape(shape, grid.full_origin())
        dp_ref_compute(
            state.ak,
            state.bk,
            state.phis,
            state.dp_ref,
            state.zs,
            rgrav,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(add=(0, 0, 1)),
        )
        # After writing, make 'dp_ref' a K-field and 'zs' an IJ-field
        state.dp_ref = utils.make_storage_data(state.dp_ref[0, 0, :], (shape[2],), (0,))
        state.zs = utils.make_storage_data(state.zs[:, :, 0], shape[0:2], (0, 0))
    n_con = get_n_con()

    # "acoustic" loop
    # called this because its timestep is usually limited by horizontal sound-wave
    # processes. Note this is often not the limiting factor near the poles, where
    # the speed of the polar night jets can exceed two-thirds of the speed of sound.
    for it in range(n_split):
        # the Lagrangian dynamics have two parts. First we advance the C-grid winds
        # by half a time step (c_sw). Then the C-grid winds are used to define advective
        # fluxes to advance the D-grid prognostic fields a full time step
        # (the rest of the routines).
        #
        # Along-surface flux terms (mass, heat, vertical momentum, vorticity,
        # kinetic energy gradient terms) are evaluated forward-in-time.
        #
        # The pressure gradient force and elastic terms are then evaluated
        # backwards-in-time, to improve stability.
        remap_step = False
        if spec.namelist.breed_vortex_inline or (it == n_split - 1):
            remap_step = True
        if not hydrostatic:
            if global_config.get_do_halo_exchange():
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
                if global_config.get_do_halo_exchange():
                    reqs["gz_quantity"] = comm.start_halo_update(
                        state.gz_quantity, n_points=utils.halo
                    )
        if it == 0:
            if global_config.get_do_halo_exchange():
                reqs["delp_quantity"].wait()
                reqs["pt_quantity"].wait()

        if it == n_split - 1 and end_step:
            if spec.namelist.use_old_omega:  # apparently True
                set_pem(
                    state.delp,
                    state.pem,
                    state.ptop,
                    origin=(grid.is_ - 1, grid.js - 1, 0),
                    domain=(grid.nic + 2, grid.njc + 2, grid.npz),
                )
        if global_config.get_do_halo_exchange():
            reqs_vector.wait()
            if not hydrostatic:
                reqs["w_quantity"].wait()

        # compute the c-grid winds at t + 1/2 timestep
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

        if spec.namelist.nord > 0 and global_config.get_do_halo_exchange():
            reqs["divgd_quantity"] = comm.start_halo_update(
                state.divgd_quantity, n_points=utils.halo
            )
        if not hydrostatic:
            if it == 0:
                if global_config.get_do_halo_exchange():
                    reqs["gz_quantity"].wait()
                copy_stencil(
                    state.gz,
                    state.zh,
                    origin=grid.full_origin(),
                    domain=grid.domain_shape_full(add=(0, 0, 1)),
                )
            else:
                copy_stencil(
                    state.zh,
                    state.gz,
                    origin=grid.full_origin(),
                    domain=grid.domain_shape_full(add=(0, 0, 1)),
                )
        if not hydrostatic:
            state.gz, state.ws3 = updatedzc.compute(
                state.dp_ref, state.zs, state.ut, state.vt, state.gz, state.ws3, dt2
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

        p_grad_c_stencil(
            grid.rdxc,
            grid.rdyc,
            state.uc,
            state.vc,
            state.delpc,
            state.pkc,
            state.gz,
            dt2,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(1, 1, 0)),
        )
        if global_config.get_do_halo_exchange():
            reqc_vector = comm.start_vector_halo_update(
                state.uc_quantity, state.vc_quantity, n_points=utils.halo
            )
            if spec.namelist.nord > 0:
                reqs["divgd_quantity"].wait()
            reqc_vector.wait()
        # use the computed c-grid winds to evolve the d-grid winds forward
        # by 1 timestep
        d_sw.compute(
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
        # note that uc and vc are not needed at all past this point.
        # they will be re-computed from scratch on the next acoustic timestep.

        if global_config.get_do_halo_exchange():
            for halovar in ["delp_quantity", "pt_quantity", "q_con_quantity"]:
                comm.halo_update(state.__getattribute__(halovar), n_points=utils.halo)

        # Not used unless we implement other betas and alternatives to nh_p_grad
        # if spec.namelist.d_ext > 0:
        #    raise 'Unimplemented namelist option d_ext > 0'
        # else:
        #    divg2 = utils.make_storage_from_shape(delz.shape, grid.compute_origin())

        if not hydrostatic:
            updatedzd.compute(
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

            if global_config.get_do_halo_exchange():
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
            if global_config.get_do_halo_exchange():
                reqs["zh_quantity"].wait()
                if grid.npx != grid.npy:
                    reqs["pkc_quantity"].wait()
            basic.multiply_constant(
                state.zh,
                state.gz,
                constants.GRAV,
                origin=(grid.is_ - 2, grid.js - 2, 0),
                domain=(grid.nic + 4, grid.njc + 4, grid.npz + 1),
            )
            if grid.npx == grid.npy and global_config.get_do_halo_exchange():
                reqs["pkc_quantity"].wait()
            if spec.namelist.beta != 0:
                raise Exception(
                    "Unimplemented namelist option -- we only support beta=0"
                )
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
            # TODO: Pass through ks, or remove, inconsistent representation vs Fortran.
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

        if global_config.get_do_halo_exchange():
            if it != n_split - 1:
                reqs_vector = comm.start_vector_halo_update(
                    state.u_quantity, state.v_quantity, n_points=utils.halo
                )
            else:
                if spec.namelist.grid_type < 4:
                    comm.synchronize_vector_interfaces(
                        state.u_quantity, state.v_quantity
                    )

    if n_con != 0 and spec.namelist.d_con > 1.0e-5:
        nf_ke = min(3, spec.namelist.nord + 1)

        if global_config.get_do_halo_exchange():
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
