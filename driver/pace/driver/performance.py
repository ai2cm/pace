import dataclasses
import os.path
import subprocess
from typing import List

import pace.util

from .report import collect_data_and_write_to_file


@dataclasses.dataclass
class PerformanceConfig:
    performance_mode: bool = False
    experiment_name: str = "test"
    timestep_timer: pace.util.Timer = pace.util.NullTimer()
    total_timer: pace.util.Timer = pace.util.NullTimer()
    times_per_step: List = dataclasses.field(default_factory=list)
    hits_per_step: List = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.performance_mode:
            self.timestep_timer = pace.util.Timer()
            self.total_timer = pace.util.Timer()

    def collect_performance(self):
        """
        Take the accumulated timings and flush them into a new entry
        in times_per_step and hits_per_step.
        """
        if self.performance_mode:
            self.times_per_step.append(self.timestep_timer.times)
            self.hits_per_step.append(self.timestep_timer.hits)
            self.timestep_timer.reset()

    def write_out_performance(
        self,
        comm,
        backend: str,
        is_orchestrated: bool,
        dt_atmos: float,
    ):
        if self.performance_mode:
            try:
                driver_path = os.path.dirname(__file__)
                git_hash = (
                    subprocess.check_output(
                        ["git", "-C", driver_path, "rev-parse", "HEAD"]
                    )
                    .decode()
                    .rstrip()
                )
            except subprocess.CalledProcessError:
                git_hash = "notarepo"

            self.times_per_step.append(self.total_timer.times)
            self.hits_per_step.append(self.total_timer.hits)
            comm.Barrier()
            while {} in self.hits_per_step:
                self.hits_per_step.remove({})
            collect_data_and_write_to_file(
                len(self.hits_per_step) - 1,
                backend,
                is_orchestrated,
                git_hash,
                comm,
                self.hits_per_step,
                self.times_per_step,
                self.experiment_name,
                dt_atmos,
            )
