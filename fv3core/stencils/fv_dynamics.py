#!/usr/bin/env python3
from types import SimpleNamespace

import fv3gfs.util as fv3util
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.del2cubed as del2cubed
import fv3core.stencils.dyn_core as dyn_core
import fv3core.stencils.moist_cv as moist_cv
import fv3core.stencils.neg_adj3 as neg_adj3
import fv3core.stencils.rayleigh_super as rayleigh_super
import fv3core.stencils.remapping as lagrangian_to_eulerian
import fv3core.stencils.tracer_2d_1l as tracer_2d_1l
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import ArgSpec, gtstencil, state_inputs
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.c2l_ord import compute_cubed_to_latlon


sd = utils.sd


@gtstencil()
def init_ph_columns(ak: sd, bk: sd, pfull: sd, ph1: sd, ph2: sd, p_ref: float):
    with computation(PARALLEL), interval(...):
        ph1 = ak + bk * p_ref
        ph2 = ak[0, 0, 1] + bk[0, 0, 1] * p_ref
        pfull = (ph2 - ph1) / log(ph2 / ph1)


@gtstencil()
def pt_adjust(pkz: sd, dp1: sd, q_con: sd, pt: sd):
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz


@gtstencil()
def set_omega(delp: sd, delz: sd, w: sd, omga: sd):
    with computation(PARALLEL), interval(...):
        omga = delp / delz * w


# TODO replace with something from fv3core.onfig probably, using the field_table
def tracers_dict(state):
    tracers = {}
    for tracername in utils.tracer_variables:
        tracers[tracername] = state.__getattribute__(tracername)
        quantity_name = utils.quantity_name(tracername)
        if quantity_name in state.__dict__:
            tracers[quantity_name] = state.__getattribute__(quantity_name)
    state.tracers = tracers


def fvdyn_temporaries(shape):
    grid = spec.grid
    tmps = {}
    halo_vars = ["cappa"]
    storage_vars = ["te_2d", "dp1", "ph1", "ph2", "dp1", "wsd"]
    column_vars = ["pfull", "gz", "cvm"]
    plane_vars = ["te_2d", "te0_2d"]
    utils.storage_dict(
        tmps,
        halo_vars + storage_vars + column_vars + plane_vars,
        shape,
        grid.default_origin(),
    )
    for q in halo_vars:
        grid.quantity_dict_update(tmps, q)
    return tmps


def compute_preamble(state, comm):
    grid = spec.grid
    init_ph_columns(
        state.ak,
        state.bk,
        state.pfull,
        state.ph1,
        state.ph2,
        spec.namelist.p_ref,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )
    if spec.namelist.hydrostatic:
        raise Exception("Hydrostatic is not implemented")
    print("FV Setup", grid.rank)
    moist_cv.fv_setup(
        state.pt,
        state.pkz,
        state.delz,
        state.delp,
        state.cappa,
        state.q_con,
        state.zvir,
        state.qvapor,
        state.qliquid,
        state.qice,
        state.qrain,
        state.qsnow,
        state.qgraupel,
        state.cvm,
        state.dp1,
    )

    if state.consv_te > 0 and not state.do_adiabatic_init:
        # NOTE not run in default configuration (turned off consv_te so we don't need a global allreduce)
        print("Compute Total Energy", grid.rank)
        moist_cv.compute_total_energy(
            state.u,
            state.v,
            state.w,
            state.delz,
            state.pt,
            state.delp,
            state.dp1,
            state.pe,
            state.peln,
            state.phis,
            state.zvir,
            state.te_2d,
            state.qvapor,
            state.qliquid,
            state.qice,
            state.qrain,
            state.qsnow,
            state.qgraupel,
        )

    if (not spec.namelist.rf_fast) and spec.namelist.tau != 0:
        if grid.grid_type < 4:
            print("Rayleigh Super", grid.rank)
            rayleigh_super.compute(
                state.u,
                state.v,
                state.w,
                state.ua,
                state.va,
                state.pt,
                state.delz,
                state.phis,
                state.bdt,
                state.ptop,
                state.pfull,
                comm,
            )
        # else:
        #     rayleigh_friction.compute()

    if spec.namelist.adiabatic and spec.namelist.kord_tm > 0:
        raise Exception(
            "unimplemented namelist options adiabatic with positive kord_tm"
        )
    else:
        print("Adjust pt", grid.rank)
        pt_adjust(
            state.pkz,
            state.dp1,
            state.q_con,
            state.pt,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )


def do_dyn(state, comm):
    grid = spec.grid
    copy_stencil(
        state.delp,
        state.dp1,
        origin=grid.default_origin(),
        domain=grid.domain_shape_standard(),
    )
    print("DynCore", grid.rank)
    dyn_core.compute(state, comm)
    if not spec.namelist.inline_q and state.nq != 0:
        if spec.namelist.z_tracer:
            print("Tracer2D1L", grid.rank)
            tracer_2d_1l.compute(
                comm,
                state.tracers,
                state.dp1,
                state.mfxd,
                state.mfyd,
                state.cxd,
                state.cyd,
                state.mdt,
                state.nq,
            )
        else:
            raise Exception("tracer_2d no =t implemented, turn on z_tracer")


def post_remap(state, comm):
    grid = spec.grid
    if not spec.namelist.hydrostatic:
        print("Omega", grid.rank)
        set_omega(
            state.delp,
            state.delz,
            state.w,
            state.omga,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )
    if spec.namelist.nf_omega > 0:
        print("Del2Cubed", grid.rank)
        comm.halo_update(state.omga_quantity, n_points=utils.halo)
        del2cubed.compute(
            state.omga, spec.namelist.nf_omega, 0.18 * grid.da_min, grid.npz
        )


def wrapup(state, comm):
    grid = spec.grid
    print("Neg Adj 3", grid.rank)
    neg_adj3.compute(
        state.qvapor,
        state.qliquid,
        state.qrain,
        state.qsnow,
        state.qice,
        state.qgraupel,
        state.qcld,
        state.pt,
        state.delp,
        state.delz,
        state.peln,
    )

    print("CubedToLatLon", grid.rank)
    compute_cubed_to_latlon(
        state.u_quantity, state.v_quantity, state.ua, state.va, comm, 1
    )


def set_constants(state):
    agrav = 1.0 / constants.GRAV
    state.rdg = -constants.RDGAS / agrav
    state.akap = constants.KAPPA
    state.dt2 = 0.5 * state.bdt

    # nq is actually given by ncnst - pnats, where those are given in atmosphere.F90 by:
    # ncnst = Atm(mytile)%ncnst
    # pnats = Atm(mytile)%flagstruct%pnats
    # here we hard-coded it because 8 is the only supported value, refactor this later!
    state.nq = 8  # state.nq_tot - spec.namelist.dnats
    state.zvir = constants.ZVIR


@state_inputs(
    ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
    ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qice", "ice_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
    ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
    ArgSpec("pt", "air_temperature", "degK", intent="inout"),
    ArgSpec("delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"),
    ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
    ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
    ArgSpec("u", "x_wind", "m/s", intent="inout"),
    ArgSpec("v", "y_wind", "m/s", intent="inout"),
    ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
    ArgSpec("ua", "x_wind_on_a_grid", "m/s", intent="inout"),
    ArgSpec("va", "y_wind_on_a_grid", "m/s", intent="inout"),
    ArgSpec("uc", "x_wind_on_c_grid", "m/s", intent="inout"),
    ArgSpec("vc", "y_wind_on_c_grid", "m/s", intent="inout"),
    ArgSpec("q_con", "total_condensate_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
    ArgSpec("phis", "surface_geopotential", "m^2 s^-2", intent="in"),
    ArgSpec(
        "pk", "interface_pressure_raised_to_power_of_kappa", "unknown", intent="inout"
    ),
    ArgSpec(
        "pkz",
        "finite_volume_mean_pressure_raised_to_power_of_kappa",
        "unknown",
        intent="inout",
    ),
    ArgSpec("ps", "surface_pressure", "Pa", intent="inout"),
    ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
    ArgSpec("ak", "atmosphere_hybrid_a_coordinate", "Pa", intent="in"),
    ArgSpec("bk", "atmosphere_hybrid_b_coordinate", "", intent="in"),
    ArgSpec("mfxd", "accumulated_x_mass_flux", "unknown", intent="inout"),
    ArgSpec("mfyd", "accumulated_y_mass_flux", "unknown", intent="inout"),
    ArgSpec("cxd", "accumulated_x_courant_number", "unknown", intent="inout"),
    ArgSpec("cyd", "accumulated_y_courant_number", "unknown", intent="inout"),
    ArgSpec(
        "diss_estd", "dissipation_estimate_from_heat_source", "unknown", intent="inout"
    ),
)
def fv_dynamics(state, comm, consv_te, do_adiabatic_init, timestep, ptop, n_split, ks):
    state.__dict__.update(
        {
            "consv_te": consv_te,
            "bdt": timestep,
            "do_adiabatic_init": do_adiabatic_init,
            "ptop": ptop,
            "n_split": n_split,
            "ks": ks,
        }
    )
    compute(state, comm)


def compute(state, comm):
    grid = spec.grid
    state.__dict__.update(fvdyn_temporaries(state.u.shape))
    set_constants(state)
    tracers_dict(state)
    last_step = False
    k_split = spec.namelist.k_split
    state.mdt = state.bdt / k_split
    compute_preamble(state, comm)
    for n_map in range(k_split):
        state.n_map = n_map + 1
        if n_map == k_split - 1:
            last_step = True
        do_dyn(state, comm)
        if grid.npz > 4:
            kord_tracer = [spec.namelist.kord_tr] * state.nq
            kord_tracer[6] = 9
            # do_omega = spec.namelist.hydrostatic and last_step
            print("Remapping", grid.rank)
            lagrangian_to_eulerian.compute(
                state.tracers,
                state.pt,
                state.delp,
                state.delz,
                state.peln,
                state.u,
                state.v,
                state.w,
                state.ua,
                state.va,
                state.cappa,
                state.q_con,
                state.pkz,
                state.pk,
                state.pe,
                state.phis,
                state.te0_2d,
                state.ps,
                state.wsd,
                state.omga,
                state.ak,
                state.bk,
                state.pfull,
                state.dp1,
                state.ptop,
                state.akap,
                state.zvir,
                last_step,
                state.consv_te,
                state.mdt,
                state.bdt,
                kord_tracer,
                state.do_adiabatic_init,
                state.nq,
            )
            if last_step:
                post_remap(state, comm)
    wrapup(state, comm)
