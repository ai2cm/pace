from typing import Mapping

from gt4py.gtscript import PARALLEL, computation, interval, log

import fv3core._config as spec
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_config as global_config
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
from fv3core.decorators import ArgSpec, FrozenStencil, get_namespace
from fv3core.stencils import tracer_2d_1l
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.c2l_ord import CubedToLatLon
from fv3core.stencils.del2cubed import HyperdiffusionDamping
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from fv3core.stencils.remapping import Lagrangian_to_Eulerian
from fv3core.utils.typing import FloatField, FloatFieldK


def pt_adjust(pkz: FloatField, dp1: FloatField, q_con: FloatField, pt: FloatField):
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz


def set_omega(delp: FloatField, delz: FloatField, w: FloatField, omga: FloatField):
    with computation(PARALLEL), interval(...):
        omga = delp / delz * w


def init_pfull(
    ak: FloatFieldK,
    bk: FloatFieldK,
    pfull: FloatField,
    p_ref: float,
):
    with computation(PARALLEL), interval(...):
        ph1 = ak + bk * p_ref
        ph2 = ak[1] + bk[1] * p_ref
        pfull = (ph2 - ph1) / log(ph2 / ph1)


def compute_preamble(
    state,
    grid,
    namelist,
    fv_setup_stencil: FrozenStencil,
    pt_adjust_stencil: FrozenStencil,
):
    if namelist.hydrostatic:
        raise NotImplementedError("Hydrostatic is not implemented")
    if __debug__:
        if grid.rank == 0:
            print("FV Setup")
    fv_setup_stencil(
        state.qvapor,
        state.qliquid,
        state.qrain,
        state.qsnow,
        state.qice,
        state.qgraupel,
        state.q_con,
        state.cvm,
        state.pkz,
        state.pt,
        state.cappa,
        state.delp,
        state.delz,
        state.dp1,
    )

    if state.consv_te > 0 and not state.do_adiabatic_init:
        raise NotImplementedError(
            "compute total energy is not implemented, it needs an allReduce"
        )

    if (not namelist.rf_fast) and namelist.tau != 0:
        raise NotImplementedError(
            "Rayleigh_Super, called when rf_fast=False and tau !=0"
        )

    if namelist.adiabatic and namelist.kord_tm > 0:
        raise NotImplementedError(
            "unimplemented namelist options adiabatic with positive kord_tm"
        )
    else:
        if __debug__:
            if grid.rank == 0:
                print("Adjust pt")
        pt_adjust_stencil(
            state.pkz,
            state.dp1,
            state.q_con,
            state.pt,
        )


def post_remap(
    state,
    comm,
    grid,
    namelist,
    hyperdiffusion: HyperdiffusionDamping,
    set_omega_stencil: FrozenStencil,
):
    grid = grid
    if not namelist.hydrostatic:
        if __debug__:
            if grid.rank == 0:
                print("Omega")
        set_omega_stencil(
            state.delp,
            state.delz,
            state.w,
            state.omga,
        )
    if namelist.nf_omega > 0:
        if __debug__:
            if grid.rank == 0:
                print("Del2Cubed")
        if global_config.get_do_halo_exchange():
            comm.halo_update(state.omga_quantity, n_points=utils.halo)
        hyperdiffusion(state.omga, 0.18 * grid.da_min)


def wrapup(
    state,
    comm: fv3gfs.util.CubedSphereCommunicator,
    grid,
    adjust_stencil: AdjustNegativeTracerMixingRatio,
    cubed_to_latlon_stencil: CubedToLatLon,
):
    if __debug__:
        if grid.rank == 0:
            print("Neg Adj 3")
    adjust_stencil(
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

    if __debug__:
        if grid.rank == 0:
            print("CubedToLatLon")
    cubed_to_latlon_stencil(
        state.u_quantity,
        state.v_quantity,
        state.ua,
        state.va,
        comm,
    )


def fvdyn_temporaries(shape, grid):
    origin = grid.full_origin()
    tmps = {}
    halo_vars = ["cappa"]
    storage_vars = ["te_2d", "dp1", "cvm"]
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

    def __init__(
        self,
        comm: fv3gfs.util.CubedSphereCommunicator,
        namelist,
        ak: fv3gfs.util.Quantity,
        bk: fv3gfs.util.Quantity,
        phis: fv3gfs.util.Quantity,
    ):
        """
        Args:
            comm: object for cubed sphere inter-process communication
            namelist: flattened Fortran namelist
            ak: atmosphere hybrid a coordinate (Pa)
            bk: atmosphere hybrid b coordinate (dimensionless)
            phis: surface geopotential height
        """
        assert namelist.moist_phys, "fvsetup is only implemented for moist_phys=true"
        assert namelist.nwat == 6, "Only nwat=6 has been implemented and tested"
        self.comm = comm
        self.grid = spec.grid
        self.namelist = namelist
        self.do_halo_exchange = global_config.get_do_halo_exchange()

        self.tracer_advection = tracer_2d_1l.TracerAdvection(comm, namelist)
        self._ak = ak.storage
        self._bk = bk.storage
        self._phis = phis.storage
        pfull_stencil = FrozenStencil(
            init_pfull, origin=(0, 0, 0), domain=(1, 1, self.grid.npz)
        )
        pfull = utils.make_storage_from_shape((1, 1, self._ak.shape[0]))
        pfull_stencil(self._ak, self._bk, pfull, self.namelist.p_ref)
        # workaround because cannot write to FieldK storage in stencil
        self._pfull = utils.make_storage_data(pfull[0, 0, :], self._ak.shape, (0,))
        self._fv_setup_stencil = FrozenStencil(
            moist_cv.fv_setup,
            externals={
                "nwat": self.namelist.nwat,
                "moist_phys": self.namelist.moist_phys,
            },
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._pt_adjust_stencil = FrozenStencil(
            pt_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._set_omega_stencil = FrozenStencil(
            set_omega,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self.acoustic_dynamics = AcousticDynamics(
            comm, namelist, self._ak, self._bk, self._pfull, self._phis
        )
        self._hyperdiffusion = HyperdiffusionDamping(self.grid, self.namelist.nf_omega)
        self._do_cubed_to_latlon = CubedToLatLon(self.grid, namelist)

        self._temporaries = fvdyn_temporaries(
            self.grid.domain_shape_full(add=(1, 1, 1)), self.grid
        )
        if not (not self.namelist.inline_q and DynamicalCore.NQ != 0):
            raise NotImplementedError("tracer_2d not implemented, turn on z_tracer")
        self._adjust_tracer_mixing_ratio = AdjustNegativeTracerMixingRatio(
            self.grid, self.namelist
        )

        self._lagrangian_to_eulerian_obj = Lagrangian_to_Eulerian(
            self.grid, namelist, DynamicalCore.NQ, self._pfull
        )

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
        tracers = {}
        for name in utils.tracer_variables[0 : DynamicalCore.NQ]:
            tracers[name] = state.__dict__[name + "_quantity"]
        tracer_storages = {name: quantity.storage for name, quantity in tracers.items()}

        state.ak = self._ak
        state.bk = self._bk
        last_step = False
        if self.do_halo_exchange:
            self.comm.halo_update(state.phis_quantity, n_points=utils.halo)
        compute_preamble(
            state,
            self.grid,
            self.namelist,
            self._fv_setup_stencil,
            self._pt_adjust_stencil,
        )

        for n_map in range(state.k_split):
            state.n_map = n_map + 1
            last_step = n_map == state.k_split - 1
            self._dyn(state, tracers, timer)

            if self.grid.npz > 4:
                # nq is actually given by ncnst - pnats,
                # where those are given in atmosphere.F90 by:
                # ncnst = Atm(mytile)%ncnst
                # pnats = Atm(mytile)%flagstruct%pnats
                # here we hard-coded it because 8 is the only supported value,
                # refactor this later!

                # do_omega = self.namelist.hydrostatic and last_step
                # TODO: Determine a better way to do this, polymorphic fields perhaps?
                # issue is that set_val in map_single expects a 3D field for the
                # "surface" array
                if __debug__:
                    if self.grid.rank == 0:
                        print("Remapping")
                with timer.clock("Remapping"):
                    self._lagrangian_to_eulerian_obj(
                        tracer_storages,
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
                        state.qcld,
                        state.pkz,
                        state.pk,
                        state.pe,
                        state.phis,
                        state.te0_2d,
                        state.ps,
                        state.wsd,
                        state.omga,
                        self._ak,
                        self._bk,
                        self._pfull,
                        state.dp1,
                        state.ptop,
                        constants.KAPPA,
                        constants.ZVIR,
                        last_step,
                        state.consv_te,
                        state.bdt / state.k_split,
                        state.bdt,
                        state.do_adiabatic_init,
                        DynamicalCore.NQ,
                    )
                if last_step:
                    post_remap(
                        state,
                        self.comm,
                        self.grid,
                        self.namelist,
                        self._hyperdiffusion,
                        self._set_omega_stencil,
                    )
        wrapup(
            state,
            self.comm,
            self.grid,
            self._adjust_tracer_mixing_ratio,
            self._do_cubed_to_latlon,
        )

    def _dyn(self, state, tracers, timer=fv3gfs.util.NullTimer()):
        copy_stencil(
            state.delp,
            state.dp1,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )
        if __debug__:
            if self.grid.rank == 0:
                print("DynCore")
        with timer.clock("DynCore"):
            self.acoustic_dynamics(state)
        if self.namelist.z_tracer:
            if __debug__:
                if self.grid.rank == 0:
                    print("TracerAdvection")
            with timer.clock("TracerAdvection"):
                self.tracer_advection(
                    tracers,
                    state.dp1,
                    state.mfxd,
                    state.mfyd,
                    state.cxd,
                    state.cyd,
                    state.mdt,
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
    dycore = utils.cached_stencil_class(DynamicalCore)(
        comm,
        spec.namelist,
        state["atmosphere_hybrid_a_coordinate"],
        state["atmosphere_hybrid_b_coordinate"],
        state["surface_geopotential"],
    )
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
