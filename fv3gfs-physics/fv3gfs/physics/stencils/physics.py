import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, exp, interval, log

import pace.dsl.gt4py_utils as utils
import pace.util
import pace.util.constants as constants
from fv3core.initialization.dycore_state import DycoreState
from fv3core.utils.grid import GridData
from fv3gfs.physics.physics_state import PhysicsState
from fv3gfs.physics.stencils.get_phi_fv3 import get_phi_fv3
from fv3gfs.physics.stencils.get_prs_fv3 import get_prs_fv3
from fv3gfs.physics.stencils.microphysics import Microphysics
from fv3gfs.physics.stencils.update_atmos_state import UpdateAtmosphereState
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, FloatField
from pace.util import TilePartitioner


def atmos_phys_driver_statein(
    prsik: FloatField,
    phii: FloatField,
    prsi: FloatField,
    delz: FloatField,
    delp: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
    pt: FloatField,
    dm: FloatField,
):
    from __externals__ import nwat, pk0inv, pktop, ptop

    with computation(BACKWARD), interval(0, -1):
        phii = phii[0, 0, 1] - delz * constants.GRAV

    with computation(PARALLEL), interval(...):
        prsik = 1.0e25
        qvapor = qvapor * delp
        qliquid = qliquid * delp
        qrain = qrain * delp
        qice = qice * delp
        qsnow = qsnow * delp
        qgraupel = qgraupel * delp
        qo3mr = qo3mr * delp
    # The following needs to execute after the above (TODO)
    with computation(PARALLEL), interval(...):
        if nwat == 6:
            delp = delp - qliquid - qrain - qice - qsnow - qgraupel

    with computation(PARALLEL), interval(0, 1):
        prsi = ptop

    with computation(FORWARD), interval(1, None):
        prsi = prsi[0, 0, -1] + delp[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        prsik = log(prsi)
        qvapor = qvapor / delp
        qliquid = qliquid / delp
        qrain = qrain / delp
        qice = qice / delp
        qsnow = qsnow / delp
        qgraupel = qgraupel / delp
        qo3mr = qo3mr / delp
        qsgs_tke = qsgs_tke / delp

    with computation(PARALLEL), interval(-1, None):
        prsik = log(prsi)

    with computation(PARALLEL), interval(0, 1):
        prsik = log(ptop)

    with computation(PARALLEL), interval(0, -1):
        qmin = 1.0e-10  # set it here since externals cannot be 2D
        qgrs_rad = max(qmin, qvapor)
        rTv = constants.RDGAS * pt * (1.0 + constants.ZVIR * qgrs_rad)
        dm = delp[0, 0, 0]
        delp = dm * rTv / (phii[0, 0, 0] - phii[0, 0, 1])
        delp = min(delp, prsi[0, 0, 1] - 0.01 * dm)
        delp = max(delp, prsi + 0.01 * dm)

    with computation(PARALLEL), interval(-1, None):
        prsik = exp(constants.KAPPA * prsik) * pk0inv

    with computation(PARALLEL), interval(0, 1):
        prsik = pktop


def prepare_microphysics(
    dz: FloatField,
    phii: FloatField,
    wmp: FloatField,
    omga: FloatField,
    qvapor: FloatField,
    pt: FloatField,
    delp: FloatField,
):
    with computation(BACKWARD), interval(...):
        dz = (phii[0, 0, 1] - phii[0, 0, 0]) * constants.RGRAV
        wmp = (
            -omga
            * (1.0 + constants.ZVIR * qvapor)
            * pt
            / delp
            * (constants.RDGAS * constants.RGRAV)
        )


@gtscript.function
def forward_euler(q_t0, q_dt, dt):
    return q_t0 + q_dt * dt


def update_physics_state_with_tendencies(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qcld: FloatField,
    pt: FloatField,
    ua: FloatField,
    va: FloatField,
    qv_dt: FloatField,
    ql_dt: FloatField,
    qr_dt: FloatField,
    qi_dt: FloatField,
    qs_dt: FloatField,
    qg_dt: FloatField,
    qa_dt: FloatField,
    pt_dt: FloatField,
    udt: FloatField,
    vdt: FloatField,
    qvapor_t1: FloatField,
    qliquid_t1: FloatField,
    qrain_t1: FloatField,
    qice_t1: FloatField,
    qsnow_t1: FloatField,
    qgraupel_t1: FloatField,
    qcld_t1: FloatField,
    pt_t1: FloatField,
    ua_t1: FloatField,
    va_t1: FloatField,
    dt: Float,
):
    with computation(PARALLEL), interval(...):
        qvapor_t1 = forward_euler(qvapor, qv_dt, dt)
        qliquid_t1 = forward_euler(qliquid, ql_dt, dt)
        qrain_t1 = forward_euler(qrain, qr_dt, dt)
        qice_t1 = forward_euler(qice, qi_dt, dt)
        qsnow_t1 = forward_euler(qsnow, qs_dt, dt)
        qgraupel_t1 = forward_euler(qgraupel, qg_dt, dt)
        qcld_t1 = forward_euler(qcld, qa_dt, dt)
        pt_t1 = forward_euler(pt, pt_dt, dt)
        ua_t1 = forward_euler(ua, udt, dt)
        va_t1 = forward_euler(va, vdt, dt)


class Physics:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        namelist,
        comm: pace.util.CubedSphereCommunicator,
        partitioner: TilePartitioner,
        rank: int,
        grid_info,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self.namelist = namelist
        origin = grid_indexing.origin_compute()
        shape = grid_indexing.domain_full(add=(1, 1, 1))
        self.setup_statein()
        self._dt_atmos = Float(self.namelist.dt_atmos)
        self._ptop = 300.0  # hard coded before we can call ak from grid: state["ak"][0]
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA

        def make_storage():
            return utils.make_storage_from_shape(
                shape, origin=origin, init=True, backend=stencil_factory.backend
            )

        self._prsi = make_storage()
        self._prsik = make_storage()
        self._dm3d = make_storage()
        self._del_gz = make_storage()
        self._full_zero_storage = make_storage()
        self._get_prs_fv3 = stencil_factory.from_origin_domain(
            func=get_prs_fv3,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._get_phi_fv3 = stencil_factory.from_origin_domain(
            func=get_phi_fv3,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._atmos_phys_driver_statein = stencil_factory.from_origin_domain(
            func=atmos_phys_driver_statein,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            externals={
                "nwat": self._nwat,
                "ptop": self._ptop,
                "pk0inv": self._pk0inv,
                "pktop": self._pktop,
            },
        )
        self._prepare_microphysics = stencil_factory.from_origin_domain(
            func=prepare_microphysics,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._update_physics_state_with_tendencies = stencil_factory.from_origin_domain(
            func=update_physics_state_with_tendencies,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._microphysics = Microphysics(stencil_factory, grid_data, namelist)
        self._update_atmos_state = UpdateAtmosphereState(
            stencil_factory, grid_data, namelist, comm, partitioner, rank, grid_info
        )

    def setup_statein(self):
        self._NQ = 8  # state.nq_tot - spec.namelist.dnats
        self._dnats = 1  # spec.namelist.dnats
        self._nwat = 6  # spec.namelist.nwat
        self._p00 = 1.0e5

    def setup_const_from_ptop(self, ptop: float):
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA

    def __call__(self, state: DycoreState, ptop: float):
        self.setup_const_from_ptop(ptop)
        physics_state = PhysicsState.from_dycore_state(state, self._full_zero_storage)
        self._atmos_phys_driver_statein(
            self._prsik,
            physics_state.phii,
            self._prsi,
            physics_state.delz,
            physics_state.delp,
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qo3mr,
            physics_state.qsgs_tke,
            physics_state.qcld,
            physics_state.pt,
            self._dm3d,
        )
        self._get_prs_fv3(
            physics_state.phii,
            self._prsi,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delprsi,
            self._del_gz,
        )
        # If PBL scheme is present, physics_state should be updated here
        self._get_phi_fv3(
            physics_state.pt,
            physics_state.qvapor,
            self._del_gz,
            physics_state.phii,
            physics_state.phil,
        )
        self._prepare_microphysics(
            physics_state.dz,
            physics_state.phii,
            physics_state.wmp,
            physics_state.omga,
            physics_state.qvapor,
            physics_state.pt,
            physics_state.delp,
        )
        microph_state = physics_state.microphysics(self._full_zero_storage)
        self._microphysics(microph_state)
        # Fortran uses IPD interface, here we use var_t1 to denote the updated field
        self._update_physics_state_with_tendencies(
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qcld,
            physics_state.pt,
            physics_state.ua,
            physics_state.va,
            microph_state.qv_dt,
            microph_state.ql_dt,
            microph_state.qr_dt,
            microph_state.qi_dt,
            microph_state.qs_dt,
            microph_state.qg_dt,
            microph_state.qa_dt,
            microph_state.pt_dt,
            microph_state.udt,
            microph_state.vdt,
            physics_state.qvapor_t1,
            physics_state.qliquid_t1,
            physics_state.qrain_t1,
            physics_state.qice_t1,
            physics_state.qsnow_t1,
            physics_state.qgraupel_t1,
            physics_state.qcld_t1,
            physics_state.pt_t1,
            physics_state.ua_t1,
            physics_state.va_t1,
            self._dt_atmos,
        )
        # [TODO]: allow update_atmos_state call when grid variables are ready
        self._update_atmos_state(state, physics_state, self._prsi)
