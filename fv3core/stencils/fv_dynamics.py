from typing import Mapping

from gt4py.gtscript import PARALLEL, computation, interval, log

import fv3core._config as spec
import fv3core.stencils.del2cubed as del2cubed
import fv3core.stencils.dyn_core as dyn_core
import fv3core.stencils.moist_cv as moist_cv
import fv3core.stencils.neg_adj3 as neg_adj3
import fv3core.stencils.rayleigh_super as rayleigh_super
import fv3core.stencils.remapping as lagrangian_to_eulerian
import fv3core.stencils.tracer_2d_1l
import fv3core.utils.global_config as global_config
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
from fv3core.decorators import ArgSpec, get_namespace, gtstencil
from fv3core.stencils import c2l_ord
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldK


@gtstencil()
def init_ph_columns(
    ak: FloatFieldK,
    bk: FloatFieldK,
    pfull: FloatField,
    p_ref: float,
):
    with computation(PARALLEL), interval(...):
        ph1 = ak + bk * p_ref
        ph2 = ak[1] + bk[1] * p_ref
        pfull = (ph2 - ph1) / log(ph2 / ph1)


@gtstencil()
def pt_adjust(pkz: FloatField, dp1: FloatField, q_con: FloatField, pt: FloatField):
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz


@gtstencil()
def set_omega(delp: FloatField, delz: FloatField, w: FloatField, omga: FloatField):
    with computation(PARALLEL), interval(...):
        omga = delp / delz * w


def compute_preamble(state, comm, grid, namelist):
    init_ph_columns(
        state.ak,
        state.bk,
        state.pfull,
        namelist.p_ref,
        origin=(0, 0, 0),
        domain=(1, 1, grid.domain_shape_compute()[2]),
    )

    state.pfull = utils.make_storage_data(state.pfull[0, 0, :], state.ak.shape, (0,))

    if namelist.hydrostatic:
        raise Exception("Hydrostatic is not implemented")
    print("FV Setup", grid.rank)
    moist_cv.fv_setup(
        state.pt,
        state.pkz,
        state.delz,
        state.delp,
        state.cappa,
        state.q_con,
        constants.ZVIR,
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
        # NOTE: Not run in default configuration (turned off consv_te so we don't
        # need a global allreduce).
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
            constants.ZVIR,
            state.te_2d,
            state.qvapor,
            state.qliquid,
            state.qice,
            state.qrain,
            state.qsnow,
            state.qgraupel,
        )

    if (not namelist.rf_fast) and namelist.tau != 0:
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

    if namelist.adiabatic and namelist.kord_tm > 0:
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


def post_remap(state, comm, grid, namelist):
    grid = grid
    if not namelist.hydrostatic:
        print("Omega", grid.rank)
        set_omega(
            state.delp,
            state.delz,
            state.w,
            state.omga,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )
    if namelist.nf_omega > 0:
        print("Del2Cubed", grid.rank)
        if global_config.get_do_halo_exchange():
            comm.halo_update(state.omga_quantity, n_points=utils.halo)
        del2cubed.compute(state.omga, namelist.nf_omega, 0.18 * grid.da_min, grid.npz)


def wrapup(state, comm: fv3gfs.util.CubedSphereCommunicator, grid):
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
    c2l_ord.compute_cubed_to_latlon(
        state.u_quantity, state.v_quantity, state.ua, state.va, comm, True
    )


def fvdyn_temporaries(shape, grid):
    origin = grid.full_origin()
    tmps = {}
    halo_vars = ["cappa"]
    storage_vars = ["te_2d", "dp1", "pfull", "cvm", "wsd_3d"]
    column_vars = ["gz"]
    plane_vars = ["te_2d", "te0_2d", "wsd"]
    utils.storage_dict(
        tmps,
        halo_vars + storage_vars,
        shape,
        origin,
    )
    utils.storage_dict(
        tmps,
        plane_vars,
        shape[0:2],
        origin[0:2],
    )
    utils.storage_dict(
        tmps,
        column_vars,
        (shape[2],),
        (origin[2],),
    )
    for q in halo_vars:
        grid.quantity_dict_update(tmps, q)
    return tmps


class DynamicalCore:
    """
    Corresponds to fv_dynamics in original Fortran sources.
    """

    # nq is actually given by ncnst - pnats, where those are given in atmosphere.F90 by:
    # ncnst = Atm(mytile)%ncnst
    # pnats = Atm(mytile)%flagstruct%pnats
    # here we hard-coded it because 8 is the only supported value, refactor this later!
    NQ = 8  # state.nq_tot - spec.namelist.dnats

    arg_specs = (
        ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
        ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
        ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
        ArgSpec("pt", "air_temperature", "degK", intent="inout"),
        ArgSpec(
            "delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"
        ),
        ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
        ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
        ArgSpec("u", "x_wind", "m/s", intent="inout"),
        ArgSpec("v", "y_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("uc", "x_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("vc", "y_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("q_con", "total_condensate_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
        ArgSpec("phis", "surface_geopotential", "m^2 s^-2", intent="in"),
        ArgSpec(
            "pk",
            "interface_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec(
            "pkz",
            "layer_mean_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec("ps", "surface_pressure", "Pa", intent="inout"),
        ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
        ArgSpec("ak", "atmosphere_hybrid_a_coordinate", "Pa", intent="in"),
        ArgSpec("bk", "atmosphere_hybrid_b_coordinate", "", intent="in"),
        ArgSpec("mfxd", "accumulated_x_mass_flux", "unknown", intent="inout"),
        ArgSpec("mfyd", "accumulated_y_mass_flux", "unknown", intent="inout"),
        ArgSpec("cxd", "accumulated_x_courant_number", "", intent="inout"),
        ArgSpec("cyd", "accumulated_y_courant_number", "", intent="inout"),
        ArgSpec(
            "diss_estd",
            "dissipation_estimate_from_heat_source",
            "unknown",
            intent="inout",
        ),
    )

    def __init__(self, comm: fv3gfs.util.CubedSphereCommunicator, namelist):
        self.comm = comm
        self.grid = spec.grid
        self.namelist = namelist
        self.do_halo_exchange = global_config.get_do_halo_exchange()
        self.tracer_advection = fv3core.stencils.tracer_2d_1l.Tracer2D1L(comm, namelist)
        # npx and npy are number of interfaces, npz is number of centers
        # and shapes should be the full data shape
        self._temporaries = fvdyn_temporaries(
            self.grid.domain_shape_full(add=(1, 1, 1)), self.grid
        )
        if not (not self.namelist.inline_q and DynamicalCore.NQ != 0):
            raise NotImplementedError("tracer_2d not implemented, turn on z_tracer")

    def step_dynamics(
        self,
        state: Mapping[str, fv3gfs.util.Quantity],
        conserve_total_energy: bool,
        do_adiabatic_init: bool,
        timestep: float,
        ptop,
        n_split: int,
        ks: int,
        timer: fv3gfs.util.Timer = fv3gfs.util.NullTimer(),
    ):
        """
        Step the model state forward by one timestep.

        Args:
            state: model prognostic state and inputs
            conserve_total_energy: if True, conserve total energy
            do_adiabatic_init: if True, do adiabatic dynamics. Used
                for model initialization.
            timestep: time to progress forward in seconds
            ptop: pressure at top of atmosphere
            n_split: number of acoustic timesteps per remapping timestep
            ks: the lowest index (highest layer) for which rayleigh friction
                and other rayleigh computations are done
            timer: if given, use for timing model execution
        """
        state = get_namespace(self.arg_specs, state)
        state.__dict__.update(
            {
                "consv_te": conserve_total_energy,
                "bdt": timestep,
                "mdt": timestep / self.namelist.k_split,
                "do_adiabatic_init": do_adiabatic_init,
                "ptop": ptop,
                "n_split": n_split,
                "k_split": self.namelist.k_split,
                "ks": ks,
            }
        )
        self._compute(state, timer)

    def _compute(
        self,
        state,
        timer: fv3gfs.util.NullTimer,
    ):
        state.__dict__.update(self._temporaries)
        last_step = False
        if self.do_halo_exchange:
            self.comm.halo_update(state.phis_quantity, n_points=utils.halo)
        compute_preamble(state, self.comm, self.grid, self.namelist)
        for n_map in range(state.k_split):
            state.n_map = n_map + 1
            if n_map == state.k_split - 1:
                last_step = True
            self._dyn(state, timer)
            if self.grid.npz > 4:
                # nq is actually given by ncnst - pnats,
                # where those are given in atmosphere.F90 by:
                # ncnst = Atm(mytile)%ncnst
                # pnats = Atm(mytile)%flagstruct%pnats
                # here we hard-coded it because 8 is the only supported value,
                # refactor this later!
                kord_tracer = [self.namelist.kord_tr] * DynamicalCore.NQ
                kord_tracer[6] = 9
                # do_omega = self.namelist.hydrostatic and last_step
                # TODO: Determine a better way to do this, polymorphic fields perhaps?
                # issue is that set_val in map_single expects a 3D field for the
                # "surface" array
                state.wsd_3d[:] = utils.reshape(state.wsd, state.wsd_3d.shape)
                print("Remapping", self.grid.rank)
                with timer.clock("Remapping"):
                    lagrangian_to_eulerian.compute(
                        state.__dict__,
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
                        state.wsd_3d,
                        state.omga,
                        state.ak,
                        state.bk,
                        state.pfull,
                        state.dp1,
                        state.ptop,
                        constants.KAPPA,
                        constants.ZVIR,
                        last_step,
                        state.consv_te,
                        state.bdt / state.k_split,
                        state.bdt,
                        kord_tracer,
                        state.do_adiabatic_init,
                        DynamicalCore.NQ,
                    )
                if last_step:
                    post_remap(state, self.comm, self.grid, self.namelist)
                state.wsd[:] = state.wsd_3d[:, :, 0]
        wrapup(state, self.comm, self.grid)

    def _dyn(self, state, timer=fv3gfs.util.NullTimer()):
        copy_stencil(
            state.delp,
            state.dp1,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )
        print("DynCore", self.grid.rank)
        with timer.clock("DynCore"):
            dyn_core.compute(state, self.comm)
        if self.namelist.z_tracer:
            print("Tracer2D1L", self.grid.rank)
            with timer.clock("TracerAdvection"):
                self.tracer_advection(
                    state.__dict__,
                    state.dp1,
                    state.mfxd,
                    state.mfyd,
                    state.cxd,
                    state.cyd,
                    state.mdt,
                    DynamicalCore.NQ,
                )


def fv_dynamics(
    state,
    comm,
    consv_te,
    do_adiabatic_init,
    timestep,
    ptop,
    n_split,
    ks,
    timer=fv3gfs.util.NullTimer(),
):
    dycore = utils.cached_stencil_class(DynamicalCore)(comm, spec.namelist)
    dycore.step_dynamics(
        state,
        consv_te,
        do_adiabatic_init,
        timestep,
        ptop,
        n_split,
        ks,
        timer,
    )
