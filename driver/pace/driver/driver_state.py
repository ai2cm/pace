from dataclasses import fields
from pace.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK, Float
from fv3gfs.physics import PhysicsState
from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from fv3core.initialization.dycore_state import DycoreState
from pace.stencils.update_atmos_state import UpdateAtmosphereState
from fv3core import DynamicalCore
from fv3gfs.physics.stencils.physics import Physics
#from stencil_functions import copy_fields_in, prepare_tendencies_and_update_tracers
#from fv_update_phys import ApplyPhysics2Dycore
import pace.util as fv3util 
from pace.dsl.stencil import FrozenStencil


class DriverState:
    # TODO, should not need all these inputs
    def __init__(self, dycore_state: DycoreState, physics_state: PhysicsState, quantity_factory, stencil_factory, namelist, comm, grid_data, grid_info):
        self.dycore_state = dycore_state
        self.physics_state = physics_state
        self.state_updater =  UpdateAtmosphereState(
            stencil_factory, grid_data, namelist, comm, grid_info, quantity_factory
        )

    def sync_state(self):
        self.state_updater(self.dycore_state, self.physics_state)
        
    @classmethod
    def init_empty(cls, grid, quantity_factory,  namelist, comm, grid_info):
         dycore_state = DycoreState.init_empty(quantity_factory)
         physics_state = PhysicsState.init_empty(quantity_factory)
         # If copy gets removed
         # physics_state = PhysicsState.init_from_dycore(quantity_factory, dycore_state)
         return cls(dycore_state, physics_state,  quantity_factory, gnamelist, comm, grid_info)


    #intended for the case of not copying from physics to dycore, reusing the same variables in the dycore and physics
    @classmethod
    def init_physics_from_dycore(cls, quantity_factory: fv3util.QuantityFactory, dycore_state: DycoreState):
        initial_storages = {}
        dycore_fields = fields(DycoreState)
        for field in fields(PhysicsState):
            if field.metadata["full_model_var"]:
                initial_storages[field.name] = getattr(dycore_state, field.name)
            else:
                initial_storages[field.name] = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  field.metadata["units"], dtype=float)
        return PhysicsState(**initial_storages, quantity_factory=quantity_factory)
 
    @classmethod
    def init_from_dycore_state(cls, dycore_state: DycoreState, quantity_factory: fv3util.QuantityFactory, stencil_factory, namelist, comm, grid_data, grid_info):
        #physics_state = PhysicsState.init_empty(quantity_factory)
        
        # If copy gets removed
        physics_state = cls.init_physics_from_dycore(quantity_factory, dycore_state)
        return cls(dycore_state, physics_state,  quantity_factory, stencil_factory, namelist, comm, grid_data,grid_info)


   
    @classmethod
    def init_from_quantities(cls, dict_of_quantities, quantity_factory: fv3util.QuantityFactory, namelist, comm, grid_info):
        dycore_state = DycoreState.init_from_quantities(dict_of_quantities)
        return cls.init_from_dycore_state(dycore_state, quantity_factory, namelist, comm, grid_info)

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
