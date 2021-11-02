from fv3gfs.util.quantity import Quantity
from fv3core.utils.typing import (
    FloatField,
    FloatFieldIJ,
    FloatFieldI,
    Float,
)
import fv3gfs.util
from fv3gfs.physics.global_constants import *
from gt4py.gtscript import (
    PARALLEL,
    FORWARD,
    BACKWARD,
    computation,
    horizontal,
    interval,
    region,
    exp,
    log,
)
import gt4py.gtscript as gtscript
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import (
    FrozenStencil,
)  # TODO: we don't want to import from fv3core
from fv3gfs.physics.stencils.update_dwind_phys import AGrid2DGridPhysics
import fv3gfs.util as fv3util
from fv3core.stencils.c2l_ord import CubedToLatLon
import fv3core.utils.global_config as global_config

# TODO: This is the same as moist_cv.py in fv3core, should move to integration dir
@gtscript.function
def moist_cvm(qvapor, gz, ql, qs):
    cvm = (1.0 - (qvapor + gz)) * cv_air + qvapor * cv_vap + ql * c_liq + qs * c_ice
    return cvm


# This is based off of moist_cv_nwat6_fn gt4py function in moist_cv.py in fv3core
def moist_cv(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    pt: FloatField,
    t_dt: FloatField,
    con_cp: Float,
    dt: Float,
):
    with computation(PARALLEL), interval(...):
        ql = qliquid + qrain
        qs = qice + qsnow + qgraupel
        gz = ql + qs
        cvm = moist_cvm(qvapor, gz, ql, qs)
        pt = pt + t_dt * dt * con_cp / cvm


def update_pressure_and_surface_winds(
    pe: FloatField,
    delp: FloatField,
    peln: FloatField,
    pk: FloatField,
    KAPPA: Float,
    ua: FloatField,
    va: FloatField,
    ps: FloatFieldIJ,
    u_srf: FloatFieldIJ,
    v_srf: FloatFieldIJ,
):

    with computation(FORWARD), interval(1, None):
        pe = pe[0, 0, -1] + delp[0, 0, -1]
    with computation(PARALLEL), interval(1, None):
        peln = log(pe[0, 0, 0])
        pk = exp(KAPPA * peln[0, 0, 0])
    with computation(FORWARD), interval(-1, None):
        ps = pe
    with computation(FORWARD), interval(-2, -1):
        u_srf = ua[0, 0, 0]
        v_srf = va[0, 0, 0]


class ApplyPhysics2Dycore:
    """
    Fortran name is fv_update_phys
    Apply the physics tendencies (u_dt, v_dt, t_dt, q_dt) consistent with 
    the FV3 discretization and definition of the prognostic variables
    """

    def __init__(
        self, grid, namelist, comm: fv3gfs.util.CubedSphereCommunicator, grid_info,
    ):
        self.grid = grid
        self.namelist = namelist
        self.comm = comm
        self._dt = Float(self.namelist.dt_atmos)
        self._moist_cv = FrozenStencil(
            moist_cv,
            origin=self.grid.compute_origin(),
            domain=self.grid.grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._update_pressure_and_surface_winds = FrozenStencil(
            update_pressure_and_surface_winds,
            origin=self.grid.compute_origin(),
            domain=self.grid.grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._AGrid2DGridPhysics = AGrid2DGridPhysics(
            self.grid, self.namelist, grid_info
        )
        self._do_cubed_to_latlon = CubedToLatLon(
            self.grid.grid_indexing, self.grid.grid_data, order=namelist.c2l_ord
        )
        origin = self.grid.grid_indexing.origin_compute()
        shape = self.grid.grid_indexing.max_shape
        full_3Dfield_1pts_halo_spec = self.grid.grid_indexing.get_quantity_halo_spec(
            shape, origin, dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], n_halo=1,
        )
        self._udt_halo_updater = self.comm.get_scalar_halo_updater(
            [full_3Dfield_1pts_halo_spec]
        )
        self._vdt_halo_updater = self.comm.get_scalar_halo_updater(
            [full_3Dfield_1pts_halo_spec]
        )
        # TODO: check if we actually need surface winds
        self._u_srf = utils.make_storage_from_shape(
            shape[0:2], origin=origin, init=True
        )
        self._v_srf = utils.make_storage_from_shape(
            shape[0:2], origin=origin, init=True
        )

    def __call__(
        self, state, u_dt: Quantity, v_dt: Quantity, t_dt: FloatField,
    ):
        self._moist_cv(
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qsnow,
            state.qice,
            state.qgraupel,
            state.pt,
            t_dt,
            cp_air,
            self._dt,
        )
        self._udt_halo_updater.start([u_dt])
        self._vdt_halo_updater.start([v_dt])
        self._update_pressure_and_surface_winds(
            state.pe,
            state.delp,
            state.peln,
            state.pk,
            KAPPA,
            state.ua,
            state.va,
            state.ps,
            self._u_srf,
            self._v_srf,
        )
        self._udt_halo_updater.wait()
        self._vdt_halo_updater.wait()
        self._AGrid2DGridPhysics(state.u, state.v, u_dt.storage, v_dt.storage)
        self._do_cubed_to_latlon(
            state.u_quantity, state.v_quantity, state.ua, state.va, self.comm,
        )
