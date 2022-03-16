from mpi4py import MPI

from fv3core._config import DynamicalCoreConfig

# TODO physics should not depend on fv3core
# but also, driver tests should not be in physics
from fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from fv3core.testing.validation import enable_selective_validation
from fv3gfs.physics import PhysicsConfig
from pace.driver.run import Driver, DriverConfig
from pace.util.namelist import Namelist


enable_selective_validation()


class TranslateDriver(TranslateFVDynamics):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        grid = grids[0]
        self.namelist: Namelist = namelist
        self.stencil_factory = stencil_factory
        self.stencil_config = self.stencil_factory.config

    def compute_parallel(self, inputs, communicator):
        dycore_state = self.state_from_inputs(inputs)
        config_info = {
            "stencil_config": self.stencil_config.stencil_kwargs,
            "initialization_type": "regression",
            "initialization_config": {"dycore_state": dycore_state, "grid": self.grid},
            "dt_atmos": self.namelist.dt_atmos,
            "diagnostics_config": {"path": "null.zarr", "names": []},
            "performance_config": {"performance_mode": False},
            "dycore_config": DynamicalCoreConfig.from_namelist(self.namelist),
            "physics_config": PhysicsConfig.from_namelist(self.namelist),
            "seconds": self.namelist.dt_atmos,
            "dycore_only": self.namelist.dycore_only,
            "nx_tile":  self.namelist.npx - 1,
            "nz" : self.namelist.npz,
            "layout": tuple(self.namelist.layout)
        }
        config = DriverConfig.from_dict(config_info)
        driver = Driver(config=config, comm=MPI.COMM_WORLD)

        driver.step_all()
        self.dycore = driver.dycore

        outputs = self.outputs_from_state(driver.state.dycore_state)
        for name, value in outputs.items():
            outputs[name] = self.subset_output(name, value)
        return outputs
