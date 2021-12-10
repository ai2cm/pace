from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
import pace.util
from fv3core.utils.grid import GridData
from fv3gfs.physics.physics_state import PhysicsState
from fv3gfs.physics.stencils.fv_update_phys import ApplyPhysics2Dycore
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, FloatField
from pace.util import TilePartitioner


def fill_gfs(pe: FloatField, q: FloatField, q_min: Float):

    with computation(BACKWARD), interval(0, -3):
        if q[0, 0, 1] < q_min:
            q = q[0, 0, 0] + (q[0, 0, 1] - q_min) * (pe[0, 0, 2] - pe[0, 0, 1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(BACKWARD), interval(1, -3):
        if q[0, 0, 0] < q_min:
            q = q_min

    with computation(FORWARD), interval(1, -2):
        if q[0, 0, -1] < 0.0:
            q = q[0, 0, 0] + q[0, 0, -1] * (pe[0, 0, 0] - pe[0, 0, -1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(FORWARD), interval(0, -2):
        if q[0, 0, 0] < 0.0:
            q = 0.0


def prepare_tendencies_and_update_tracers(
    u_dt: FloatField,
    v_dt: FloatField,
    pt_dt: FloatField,
    u_t1: FloatField,
    v_t1: FloatField,
    pt_t1: FloatField,
    qvapor_t1: FloatField,
    qliquid_t1: FloatField,
    qrain_t1: FloatField,
    qsnow_t1: FloatField,
    qice_t1: FloatField,
    qgraupel_t1: FloatField,
    u_t0: FloatField,
    v_t0: FloatField,
    pt_t0: FloatField,
    qvapor_t0: FloatField,
    qliquid_t0: FloatField,
    qrain_t0: FloatField,
    qsnow_t0: FloatField,
    qice_t0: FloatField,
    qgraupel_t0: FloatField,
    prsi: FloatField,
    delp: FloatField,
    rdt: Float,
):
    """Gather tendencies and adjust dycore tracers values
    GFS total air mass = dry_mass + water_vapor (condensate excluded)
    GFS mixing ratios  = tracer_mass / (dry_mass + vapor_mass)
    FV3 total air mass = dry_mass + [water_vapor + condensate ]
    FV3 mixing ratios  = tracer_mass / (dry_mass+vapor_mass+cond_mass)
    """
    with computation(PARALLEL), interval(0, -1):
        u_dt += (u_t1 - u_t0) * rdt
        v_dt += (v_t1 - v_t0) * rdt
        pt_dt += (pt_t1 - pt_t0) * rdt
        dp = prsi[0, 0, 1] - prsi[0, 0, 0]
        qwat_qv = dp * qvapor_t1
        qwat_ql = dp * qliquid_t1
        qwat_qr = dp * qrain_t1
        qwat_qs = dp * qsnow_t1
        qwat_qi = dp * qice_t1
        qwat_qg = dp * qgraupel_t1
        qt = qwat_qv + qwat_ql + qwat_qr + qwat_qs + qwat_qi + qwat_qg
        q_sum = qvapor_t0 + qliquid_t0 + qrain_t0 + qsnow_t0 + qice_t0 + qgraupel_t0
        q0 = delp * (1.0 - q_sum) + qt
        delp = q0
        qvapor_t0 = qwat_qv / q0
        qliquid_t0 = qwat_ql / q0
        qrain_t0 = qwat_qr / q0
        qsnow_t0 = qwat_qs / q0
        qice_t0 = qwat_qi / q0
        qgraupel_t0 = qwat_qg / q0


class UpdateAtmosphereState:
    """Fortran name is atmosphere_state_update
    This is an API to apply tendencies and compute a consistent prognostic state.
    """

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
        self._rdt = 1.0 / Float(self.namelist.dt_atmos)
        self._fill_GFS = stencil_factory.from_origin_domain(
            fill_gfs,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._prepare_tendencies_and_update_tracers = (
            stencil_factory.from_origin_domain(
                prepare_tendencies_and_update_tracers,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            )
        )

        def make_storage():
            return utils.make_storage_from_shape(
                shape, origin=origin, init=True, backend=stencil_factory.backend
            )

        self._u_dt = make_storage()
        self._v_dt = make_storage()
        self._pt_dt = make_storage()
        self._apply_physics2dycore = ApplyPhysics2Dycore(
            stencil_factory,
            grid_data,
            self.namelist,
            comm,
            partitioner,
            rank,
            grid_info,
        )

    def __call__(
        self,
        dycore_state,
        phy_state: PhysicsState,
        prsi: FloatField,  # consider pushing this into physics_state
    ):
        self._fill_GFS(prsi, phy_state.qvapor_t1, 1.0e-9)
        self._prepare_tendencies_and_update_tracers(
            self._u_dt,
            self._v_dt,
            self._pt_dt,
            phy_state.ua_t1,
            phy_state.va_t1,
            phy_state.pt_t1,
            phy_state.qvapor_t1,
            phy_state.qliquid_t1,
            phy_state.qrain_t1,
            phy_state.qsnow_t1,
            phy_state.qice_t1,
            phy_state.qgraupel_t1,
            phy_state.ua,
            phy_state.va,
            phy_state.pt,
            dycore_state.qvapor,
            dycore_state.qliquid,
            dycore_state.qrain,
            dycore_state.qsnow,
            dycore_state.qice,
            dycore_state.qgraupel,
            prsi,
            dycore_state.delp,
            self._rdt,
        )
        self._apply_physics2dycore(
            dycore_state,
            self._u_dt,
            self._v_dt,
            self._pt_dt,
        )
