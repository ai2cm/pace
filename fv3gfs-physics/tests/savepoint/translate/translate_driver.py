import pace.dsl
import pace.util
from fv3core._config import DynamicalCoreConfig

# TODO physics should not depend on fv3core
# but also, driver tests should not be in physics
from fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from fv3core.testing.validation import enable_selective_validation
from fv3gfs.physics import PhysicsConfig, PhysicsState
from pace.driver.run import Driver, DriverConfig
from pace.driver.state import TendencyState
from pace.util.namelist import Namelist


enable_selective_validation()


class TranslateDriver(TranslateFVDynamics):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.namelist: Namelist = namelist
        self.stencil_factory = stencil_factory
        self.stencil_config = self.stencil_factory.config

    def compute_parallel(self, inputs, communicator):
        dycore_state = self.state_from_inputs(inputs)
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=pace.util.N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=self.namelist.layout,
            tile_partitioner=communicator.partitioner.tile,
            tile_rank=communicator.tile.rank,
        )

        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, backend=self.stencil_config.compilation_config.backend
        )
        physics_state = PhysicsState.init_zeros(
            quantity_factory=quantity_factory, active_packages=["microphysics"]
        )
        tendency_state = TendencyState.init_zeros(
            quantity_factory=quantity_factory,
        )
        config_info = {
            "stencil_config": self.stencil_config,
            "initialization": {
                "type": "predefined",
                "config": {
                    "dycore_state": dycore_state,
                    "grid_data": self.grid.grid_data,
                    "damping_coefficients": self.grid.damping_coefficients,
                    "driver_grid_data": self.grid.driver_grid_data,
                    "physics_state": physics_state,
                    "tendency_state": tendency_state,
                },
            },
            "dt_atmos": self.namelist.dt_atmos,
            "diagnostics_config": {"path": "null.zarr", "names": []},
            "performance_config": {"performance_mode": False},
            "dycore_config": DynamicalCoreConfig.from_namelist(self.namelist),
            "physics_config": PhysicsConfig.from_namelist(self.namelist),
            "seconds": self.namelist.dt_atmos,
            "dycore_only": self.namelist.dycore_only,
            "nx_tile": self.namelist.npx - 1,
            "nz": self.namelist.npz,
            "layout": tuple(self.namelist.layout),
        }
        config = DriverConfig.from_dict(config_info)
        driver = Driver(config=config)

        driver.step_all()
        self.dycore = driver.dycore

        outputs = self.outputs_from_state(driver.state.dycore_state)
        for name, value in outputs.items():
            outputs[name] = self.subset_output(name, value)
        return outputs
