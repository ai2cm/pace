from dataclasses import dataclass, field, fields, InitVar
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK, Float
import copy
import fv3core
from fv3gfs.physics.physics_state import PhysicsState
from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from fv3core.dycore_state import DycoreState
from fv3core import DynamicalCore
from fv3gfs.physics.stencils.physics import Physics
from stencil_functions import copy_fields_in, fill_gfs, prepare_tendencies_and_update_tracers
from fv_update_phys import ApplyPhysics2Dycore
import fv3gfs.util as fv3util 
from fv3core.decorators import FrozenStencil


class ModelState:
    # TODO, should not need all these inputs
    def __init__(self, dycore_state: DycoreState, physics_state: PhysicsState, quantity_factory, grid, namelist, comm, grid_info):
        self.dycore_state = dycore_state
        self.physics_state = physics_state
        self.state_updater = UpdateAtmosphereState(
            grid, namelist, comm, grid_info, quantity_factory
        )
   
    @classmethod
    def init_empty(cls, grid, quantity_factory,  namelist, comm, grid_info):
         dycore_state = DycoreState.init_empty(quantity_factory)
         physics_state = PhysicsState.init_empty(quantity_factory)
         # If copy gets removed
         # physics_state = PhysicsState.init_from_dycore(quantity_factory, dycore_state)
         return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)
    
    @classmethod
    def init_from_serialized_data(cls, serializer, grid, quantity_factory: fv3util.QuantityFactory, namelist, comm, grid_info):
        dycore_state = DycoreState.init_from_serialized_data(serializer, grid, quantity_factory)
        physics_state = PhysicsState.init_empty(quantity_factory)
        return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)

    @classmethod
    def init_from_numpy_arrays(cls, dict_of_numpy_arrays, grid, quantity_factory: fv3util.QuantityFactory, namelist, comm, grid_info):
        dycore_state = DycoreState.init_numpy_arrays(dict_of_numpy_arrays, quantity_factory)
        physics_state = PhysicsState.init_empty(quantity_factory)
        return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)

    @classmethod
    def init_from_dycore_state(cls, dycore_state: DycoreState, grid, quantity_factory: fv3util.QuantityFactory, namelist, comm, grid_info):
        physics_state = PhysicsState.init_empty(quantity_factory)
        # If copy gets removed
        # physics_state = PhysicsState.init_from_dycore(quantity_factory, dycore_state)
        return cls(dycore_state, physics_state,  quantity_factory, grid, namelist, comm, grid_info)
    
    def update_physics_inputs_state(self):
        self.state_updater.copy_from_dycore_to_physics(self.dycore_state, self.physics_state)

    def update_dycore_state(self):
        self.state_updater(self.dycore_state, self.physics_state)

    def step_physics(self, physics: Physics):
        self.update_physics_inputs_state()
        physics(self.physics_state)
        self.update_dycore_state()

    def step_dynamics(self, dycore: DynamicalCore, do_adiabatic_init: bool, bdt: float):
        dycore.step_dynamics(
            self.dycore_state,
            do_adiabatic_init,
            bdt,  
        )

class UpdateAtmosphereState:
    """Fortran name is atmosphere_state_update
    This is an API to apply tendencies and compute a consistent prognostic state.
    """

    def __init__(
            self, grid, namelist, comm: fv3util.CubedSphereCommunicator, grid_info, quantity_factory: fv3util.QuantityFactory
    ):
        self.grid = grid
        self.namelist = namelist
        origin = self.grid.compute_origin()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._rdt = 1.0 / Float(self.namelist.dt_atmos)
        self._fill_GFS = FrozenStencil(
            fill_gfs,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._prepare_tendencies_and_update_tracers = FrozenStencil(
            prepare_tendencies_and_update_tracers,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._copy_fields_in = FrozenStencil(
            func=copy_fields_in,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(1, 1, 1))
        )
        self._u_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "m/s^2", dtype=float)
        self._v_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "m/s^2", dtype=float) 
        self._pt_dt = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "degK/s", dtype=float) 
        self._apply_physics2dycore = ApplyPhysics2Dycore(
            self.grid, self.namelist, comm, grid_info
        )
        

    def __call__(
        self,
        dycore_state: DycoreState,
        phy_state: PhysicsState,
    ):
        self._fill_GFS(phy_state.prsi, phy_state.qvapor_t1, 1.0e-9)
        self._prepare_tendencies_and_update_tracers(
            self._u_dt.storage,
            self._v_dt.storage,
            self._pt_dt.storage,
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
            phy_state.prsi,
            dycore_state.delp,
            self._rdt,
        )
        self._apply_physics2dycore(
            dycore_state, self._u_dt, self._v_dt, self._pt_dt.storage,
        )

    # TODO -- we can probably use the dycore state and not copy
    # check if any of these are modified internally that
    # are fields that are not then overwritten as a last step
    def copy_from_dycore_to_physics(self, dycore_state, physics_state):
        self._copy_fields_in(
            dycore_state.qvapor,
            dycore_state.qliquid,
            dycore_state.qrain,
            dycore_state.qice,
            dycore_state.qsnow,
            dycore_state.qgraupel,
            dycore_state.qo3mr,
            dycore_state.qsgs_tke,
            dycore_state.qcld,
            dycore_state.pt,
            dycore_state.delp,
            dycore_state.delz,
            dycore_state.ua,
            dycore_state.va,
            dycore_state.w,
            dycore_state.omga,
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
            physics_state.delp,
            physics_state.delz,
            physics_state.ua,
            physics_state.va,
            physics_state.w,
            physics_state.omga,
        )
