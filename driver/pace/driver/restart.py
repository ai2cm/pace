import dataclasses
import os
from datetime import datetime
from typing import List

import yaml

from .driver import DriverConfig
from .state import DriverState


@dataclasses.dataclass()
class Restart:
    save_restart: bool = False
    intermediate_restart: List[int] = dataclasses.field(default_factory=list)
    save_intermediate_restart: bool = False

    def __post_init__(self):
        if len(self.intermediate_restart) > 0:
            self.save_intermediate_restart = True

    def save_state_as_restart(self, state: DriverState, comm, restart_path: str):
        state.save_state(comm=comm, restart_path=restart_path)

    def write_restart_config(
        self,
        comm,
        time: datetime,
        driver_config: DriverConfig,
        restart_path: str,
    ):
        if comm.Get_rank() == 0:
            config_dict = dataclasses.asdict(driver_config)
            if driver_config.stencil_config.dace_config:
                config_dict["stencil_config"][
                    "dace_config"
                ] = driver_config.stencil_config.dace_config.as_dict()
            config_dict["stencil_config"][
                "compilation_config"
            ] = driver_config.stencil_config.compilation_config.as_dict()
            config_dict["performance_config"].pop("times_per_step", None)
            config_dict["performance_config"].pop("hits_per_step", None)
            config_dict["performance_config"].pop("timestep_timer", None)
            config_dict["performance_config"].pop("total_timer", None)
            for field in ["dt_atmos", "layout", "npx", "npy", "npz", "ntiles"]:
                config_dict["dycore_config"].pop(field, None)
                config_dict["physics_config"].pop(field, None)
            config_dict["initialization"]["type"] = "restart"
            config_dict["initialization"]["config"]["start_time"] = time
            config_dict["initialization"]["config"][
                "path"
            ] = f"{os.getcwd()}/{restart_path}"
            with open(f"{restart_path}/restart.yaml", "w") as file:
                yaml.safe_dump(config_dict, file)
