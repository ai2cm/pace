from typing import Optional

from gt4py.gtscript import PARALLEL, computation, interval, log

import fv3core.stencils.moist_cv as moist_cv
import pace.dsl.gt4py_utils as utils
import pace.util
import pace.util.constants as constants
from fv3core._config import DynamicalCoreConfig
from fv3core.initialization.dycore_state import DycoreState
from fv3core.stencils import fvtp2d, tracer_2d_1l
from fv3core.stencils.basic_operations import copy_defn
from fv3core.stencils.del2cubed import HyperdiffusionDamping
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from fv3core.stencils.remapping import LagrangianToEulerian
from pace.dsl.stencil import FrozenStencil, StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from pace.stencils.c2l_ord import CubedToLatLon
from pace.util.grid import DampingCoefficients, GridData
from pace.util.halo_updater import HaloUpdater


# nq is actually given by ncnst - pnats, where those are given in atmosphere.F90 by:
# ncnst = Atm(mytile)%ncnst
# pnats = Atm(mytile)%flagstruct%pnats
# here we hard-coded it because 8 is the only supported value, refactor this later!
NQ = 8  # state.nq_tot - spec.namelist.dnats


def pt_adjust(pkz: FloatField, dp1: FloatField, q_con: FloatField, pt: FloatField):
    """
    Args:
        pkz (in):
        dp1 (in):
        q_con (in):
        pt (out):
    """
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz


def set_omega(delp: FloatField, delz: FloatField, w: FloatField, omga: FloatField):
    """
    Args:
        delp (in):
        delz (in):
        w (in):
        omga (out):
    """
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
    is_root_rank: bool,
    config: DynamicalCoreConfig,
    fv_setup_stencil: FrozenStencil,
    pt_adjust_stencil: FrozenStencil,
):
    if config.hydrostatic:
        raise NotImplementedError("Hydrostatic is not implemented")
    if __debug__:
        if is_root_rank:
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

    if (not config.rf_fast) and config.tau != 0:
        raise NotImplementedError(
            "Rayleigh_Super, called when rf_fast=False and tau !=0"
        )

    if config.adiabatic and config.kord_tm > 0:
        raise NotImplementedError(
            "unimplemented namelist options adiabatic with positive kord_tm"
        )
    else:
        if __debug__:
            if is_root_rank:
                print("Adjust pt")
        pt_adjust_stencil(
            state.pkz,
            state.dp1,
            state.q_con,
            state.pt,
        )


def post_remap(
    state: DycoreState,
    is_root_rank: bool,
    config: DynamicalCoreConfig,
    hyperdiffusion: HyperdiffusionDamping,
    set_omega_stencil: FrozenStencil,
    omega_halo_updater: HaloUpdater,
    da_min: FloatFieldIJ,
):
    if not config.hydrostatic:
        if __debug__:
            if is_root_rank:
                print("Omega")
        set_omega_stencil(
            state.delp,
            state.delz,
            state.w,
            state.omga,
        )
    if config.nf_omega > 0:
        if __debug__:
            if is_root_rank == 0:
                print("Del2Cubed")
        omega_halo_updater.update([state.omga])
        hyperdiffusion(state.omga, 0.18 * da_min)


def wrapup(
    state: DycoreState,
    comm: pace.util.CubedSphereCommunicator,
    adjust_stencil: AdjustNegativeTracerMixingRatio,
    cubed_to_latlon_stencil: CubedToLatLon,
    is_root_rank: bool,
):
    if __debug__:
        if is_root_rank:
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
        if is_root_rank:
            print("CubedToLatLon")
    cubed_to_latlon_stencil(
        state.u,
        state.v,
        state.ua,
        state.va,
        comm,
    )


def fvdyn_temporaries(quantity_factory: pace.util.QuantityFactory):
    tmps = {}
    for name in ["te_2d", "te0_2d", "wsd"]:
        quantity = quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
        )
        tmps[name] = quantity
    for name in ["cappa", "dp1", "cvm"]:
        quantity = quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            units="unknown",
        )
        tmps[name] = quantity
    return tmps


class DynamicalCore:
    """
    Corresponds to fv_dynamics in original Fortran sources.
    """

    def __init__(
        self,
        comm: pace.util.CubedSphereCommunicator,
        grid_data: GridData,
        stencil_factory: StencilFactory,
        damping_coefficients: DampingCoefficients,
        config: DynamicalCoreConfig,
        phis: pace.util.Quantity,
        checkpointer: Optional[pace.util.Checkpointer] = None,
    ):
        """
        Args:
            comm: object for cubed sphere inter-process communication
            grid_data: metric terms defining the model grid
            stencil_factory: creates stencils
            damping_coefficients: damping configuration/constants
            config: configuration of dynamical core, for example as would be set by
                the namelist in the Fortran model
            phis: surface geopotential height
            checkpointer: if given, used to perform operations on model data
                at specific points in model execution, such as testing against
                reference data
        """
        # nested and stretched_grid are options in the Fortran code which we
        # have not implemented, so they are hard-coded here.
        self.call_checkpointer = checkpointer is not None
        self.checkpointer = checkpointer
        nested = False
        stretched_grid = False
        grid_indexing = stencil_factory.grid_indexing
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=config.npx - 1,
            ny_tile=config.npy - 1,
            nz=config.npz,
            n_halo=grid_indexing.n_halo,
            layout=config.layout,
            tile_partitioner=comm.tile.partitioner,
            tile_rank=comm.tile.rank,
            extra_dim_lengths={},
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, backend=stencil_factory.backend
        )
        assert config.moist_phys, "fvsetup is only implemented for moist_phys=true"
        assert config.nwat == 6, "Only nwat=6 has been implemented and tested"
        self.comm = comm
        self.grid_data = grid_data
        self.grid_indexing = grid_indexing
        self._da_min = damping_coefficients.da_min
        self.config = config

        tracer_transport = fvtp2d.FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_tr,
        )
        self.tracer_advection = tracer_2d_1l.TracerAdvection(
            stencil_factory, tracer_transport, self.grid_data, comm, NQ
        )
        self._ak = grid_data.ak
        self._bk = grid_data.bk
        self._phis = phis
        self._ptop = self.grid_data.ptop
        pfull_stencil = stencil_factory.from_origin_domain(
            init_pfull, origin=(0, 0, 0), domain=(1, 1, grid_indexing.domain[2])
        )
        pfull = utils.make_storage_from_shape(
            (1, 1, self._ak.shape[0]), backend=stencil_factory.backend
        )
        pfull_stencil(self._ak, self._bk, pfull, self.config.p_ref)
        # workaround because cannot write to FieldK storage in stencil
        self._pfull = utils.make_storage_data(
            pfull[0, 0, :], self._ak.shape, (0,), backend=stencil_factory.backend
        )
        self._fv_setup_stencil = stencil_factory.from_origin_domain(
            moist_cv.fv_setup,
            externals={
                "nwat": self.config.nwat,
                "moist_phys": self.config.moist_phys,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._pt_adjust_stencil = stencil_factory.from_origin_domain(
            pt_adjust,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._set_omega_stencil = stencil_factory.from_origin_domain(
            set_omega,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._copy_stencil = stencil_factory.from_origin_domain(
            copy_defn,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
        )
        self.acoustic_dynamics = AcousticDynamics(
            comm,
            stencil_factory,
            grid_data,
            damping_coefficients,
            config.grid_type,
            nested,
            stretched_grid,
            self.config.acoustic_dynamics,
            self._pfull,
            self._phis,
            checkpointer=checkpointer,
        )
        self._hyperdiffusion = HyperdiffusionDamping(
            stencil_factory,
            damping_coefficients,
            grid_data.rarea,
            self.config.nf_omega,
        )
        self._cubed_to_latlon = CubedToLatLon(
            stencil_factory, grid_data, order=config.c2l_ord
        )

        self._temporaries = fvdyn_temporaries(quantity_factory)
        if not (not self.config.inline_q and NQ != 0):
            raise NotImplementedError("tracer_2d not implemented, turn on z_tracer")
        self._adjust_tracer_mixing_ratio = AdjustNegativeTracerMixingRatio(
            stencil_factory,
            self.config.check_negative,
            self.config.hydrostatic,
        )

        self._lagrangian_to_eulerian_obj = LagrangianToEulerian(
            stencil_factory,
            config.remapping,
            grid_data.area_64,
            NQ,
            self._pfull,
        )

        full_xyz_spec = grid_indexing.get_quantity_halo_spec(
            grid_indexing.domain_full(add=(1, 1, 1)),
            grid_indexing.origin_compute(),
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            n_halo=utils.halo,
            backend=stencil_factory.backend,
        )
        self._omega_halo_updater = self.comm.get_scalar_halo_updater([full_xyz_spec])

    def _checkpoint_fvdynamics(self, state: DycoreState, tag: str):
        if self.call_checkpointer:
            self.checkpointer(
                f"FVDynamics-{tag}",
                u=state.u,
                v=state.v,
                w=state.w,
                delz=state.delz,
                ua=state.ua,
                va=state.va,
                uc=state.uc,
                vc=state.vc,
                qvapor=state.qvapor,
            )

    def step_dynamics(
        self,
        state: DycoreState,
        conserve_total_energy: float,
        do_adiabatic_init: bool,
        timestep: float,
        n_split: int,
        timer: pace.util.Timer = pace.util.NullTimer(),
    ):
        """
        Step the model state forward by one timestep.

        Args:
            state: model prognostic state and inputs
            conserve_total_energy: if True, conserve total energy
            do_adiabatic_init: if True, do adiabatic dynamics. Used
                for model initialization.
            timestep: time to progress forward in seconds
            n_split: number of acoustic timesteps per remapping timestep
            timer: if given, use for timing model execution
        """
        self._checkpoint_fvdynamics(state=state, tag="In")
        # TODO: state should be a statically typed class, move these to the
        # definition of DycoreState and pass them on init or alternatively
        # move these to/get these from the namelist/configuration class
        state.__dict__.update(
            {
                "consv_te": conserve_total_energy,
                "bdt": timestep,
                "mdt": timestep / self.config.k_split,
                "do_adiabatic_init": do_adiabatic_init,
                "n_split": n_split,
                "k_split": self.config.k_split,
            }
        )
        self._compute(state, timer)
        self._checkpoint_fvdynamics(state=state, tag="Out")

    # TODO: type hint state when it is possible to do so, when it is a static type
    def _compute(
        self,
        state,
        timer: pace.util.Timer,
    ):
        # TODO: put temporaries on a statically typed container class, as they are not
        # attributes of DycoreState
        state.__dict__.update(self._temporaries)
        tracers = {}
        for name in utils.tracer_variables[0:NQ]:
            tracers[name] = state.__dict__[name]
        tracer_storages = {name: quantity.storage for name, quantity in tracers.items()}

        # TODO: ak and bk are not attributes of DycoreState, put them on a statically
        # typed class that has them as attributes
        state.ak = self._ak
        state.bk = self._bk
        last_step = False
        compute_preamble(
            state,
            is_root_rank=self.comm.rank == 0,
            config=self.config,
            fv_setup_stencil=self._fv_setup_stencil,
            pt_adjust_stencil=self._pt_adjust_stencil,
        )

        for n_map in range(state.k_split):
            state.n_map = n_map + 1
            last_step = n_map == state.k_split - 1
            self._dyn(state, tracers, timer)

            if self.grid_indexing.domain[2] > 4:
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
                    if self.comm.rank == 0:
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
                        self._ptop,
                        constants.KAPPA,
                        constants.ZVIR,
                        last_step,
                        state.consv_te,
                        state.bdt / state.k_split,
                        state.bdt,
                        state.do_adiabatic_init,
                        NQ,
                    )
                if last_step:
                    post_remap(
                        state,
                        is_root_rank=self.comm.rank == 0,
                        config=self.config,
                        hyperdiffusion=self._hyperdiffusion,
                        set_omega_stencil=self._set_omega_stencil,
                        omega_halo_updater=self._omega_halo_updater,
                        da_min=self._da_min,
                    )
        wrapup(
            state,
            comm=self.comm,
            adjust_stencil=self._adjust_tracer_mixing_ratio,
            cubed_to_latlon_stencil=self._cubed_to_latlon,
            is_root_rank=self.comm.rank == 0,
        )

    # TODO: type hint state when it is possible to do so, when it is a static type
    def _dyn(self, state, tracers, timer=pace.util.NullTimer()):
        self._copy_stencil(
            state.delp,
            state.dp1,
        )
        if __debug__:
            if self.comm.rank == 0:
                print("DynCore")
        with timer.clock("DynCore"):
            self.acoustic_dynamics(state)
        if self.config.z_tracer:
            if __debug__:
                if self.comm.rank == 0:
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
