from fv3gfs.physics.physics_state import PhysicsState
from fv3gfs.physics.stencils.get_phi_fv3 import get_phi_fv3
from fv3gfs.physics.stencils.get_prs_fv3 import get_prs_fv3
from fv3gfs.physics.stencils.microphysics import Microphysics, MicrophysicsState
from fv3gfs.physics.global_constants import *
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField
from fv3core.decorators import FrozenStencil
from gt4py.gtscript import PARALLEL, computation
from gt4py.gtscript import (
    PARALLEL,
    FORWARD,
    BACKWARD,
    computation,
    horizontal,
    interval,
)

# TODO: stencil not completed yet
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
):
    from __externals__ import nwat, ptop

    with computation(BACKWARD), interval(0, -1):
        phii = phii[0, 0, 1] - delz * grav

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


class Physics:
    def __init__(self, grid, namelist):
        self.grid = grid
        self.namelist = namelist
        origin = self.grid.compute_origin()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._phii = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._psri = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._prsik = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._dm = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._del = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._del_gz = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._phil = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._dz = utils.make_storage_from_shape(shape, origin=origin, init=True)

        self._get_prs_fv3 = FrozenStencil(
            func=get_prs_fv3,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(),
        )
        self._get_phi_fv3 = FrozenStencil(
            func=get_phi_fv3,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(),
        )

        self._microphysics = Microphysics(grid, namelist)

    def setup_statein(self):
        self._NQ = 8  # state.nq_tot - spec.namelist.dnats
        self._dnats = 1  # spec.namelist.dnats
        self._nwat = 6  # spec.namelist.nwat
        self._p00 = 1.0e5

    def setup_const_from_state(self, state: dict):
        self._ptop = state["ak"][0]
        self._pktop = (self._ptop / self._p00) ** KAPPA
        self._pk0inv = (1.0 / self._p00) ** KAPPA

    def prepare_physics_state(self, phy_state: PhysicsState):
        pass

    def __call__(self, state: dict):
        self.setup_const_from_state(state)
        phy = PhysicsState(state, self.grid)
        physics_state = phy.physics_state
        self._get_prs_fv3(
            self._phii,
            self._prsi,
            physics_state.pt,
            physics_state.qvapor,
            self._del,
            self._del_gz,
        )
        # If PBL is present, physics_state should be updated here
        self._get_phi_fv3(
            physics_state.pt, physics_state.qvapor, self._del_gz, self._phii, self._phil
        )

        for k in range(1, self.grid.npz + 1):  # (TODO) check if it goes from 1
            self._dz[:, :, k] = (self._phii[:, :, k] - self._phii[:, :, k - 1]) * rgrav

        self._microphysics(phy.microphysics)
