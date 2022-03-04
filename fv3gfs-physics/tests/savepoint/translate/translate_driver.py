import dataclasses

import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.stencils.fv_update_phys import ApplyPhysics2Dycore
from pace.stencils.testing.parallel_translate import ParallelTranslate2Py
from pace.util.grid import DriverGridData
from pace.driver.run import DriverConfig, Driver
# TODO physics should not depend on fv3core
# but also, driver tests should not be in physics
from fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from fv3core._config import DynamicalCoreConfig
from fv3gfs.physics import PhysicsConfig
from pace.util.namelist import Namelist
from mpi4py import MPI
import pace.dsl.gt4py_utils as utils
from fv3core.testing.validation import enable_selective_validation
enable_selective_validation()
# TODO 
class TranslateDriver(TranslateFVDynamics):
   
   
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        grid = grids[0]
        self.namelist: Namelist = namelist
        self.stencil_factory = stencil_factory
        self.stencil_config = self.stencil_factory.config
        self.max_error = 1e-5


   

    def compute_parallel(self, inputs, communicator):
        dycore_state = self.state_from_inputs(inputs)
        config_info = {
             "stencil_config": self.stencil_config.stencil_kwargs,
             "initialization_type": "regression",
             "initialization_config": {
                  "dycore_state": dycore_state,
                  "grid": self.grid
             },
             "dt_atmos": self.namelist.dt_atmos,
             "diagnostics_config": {
                  "path": "null.zarr",
                  "names": []
             },
            "dycore_config":  DynamicalCoreConfig.from_namelist(self.namelist),
            "physics_config": PhysicsConfig.from_namelist(self.namelist),
            "seconds": self.namelist.dt_atmos,
            "dycore_only": self.namelist.dycore_only,
        }
        config=DriverConfig.from_dict(config_info)
        driver = Driver(config=config, comm=MPI.COMM_WORLD)
       
        driver.step_all()
        self.dycore = driver.dycore
     
        outputs = self.outputs_from_state(driver.state.dycore_state)
        for name, value in outputs.items():
            outputs[name] = self.subset_output(name, value)
        return outputs

    #def subset_output(self, varname, output):
    #    return output


    
