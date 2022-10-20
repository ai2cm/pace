import os.path
import subprocess
from typing import List, Mapping, Protocol

import pace.util

from .report import collect_data_and_write_to_file


class AbstractPerformanceCollector(Protocol):
    total_timer: pace.util.Timer
    timestep_timer: pace.util.Timer

    def collect_performance(self):
        ...

    def write_out_performance(
        self,
        backend: str,
        is_orchestrated: bool,
        dt_atmos: float,
    ):
        ...


class PerformanceCollector:
    def __init__(self, experiment_name: str, comm: pace.util.Comm):
        self.times_per_step: List[Mapping[str, float]] = []
        self.hits_per_step: List[Mapping[str, float]] = []
        self.timestep_timer = pace.util.Timer()
        self.total_timer = pace.util.Timer()
        self.experiment_name = experiment_name
        self.comm = comm

    def collect_performance(self):
        """
        Take the accumulated timings and flush them into a new entry
        in times_per_step and hits_per_step.
        """
        self.times_per_step.append(self.timestep_timer.times)
        self.hits_per_step.append(self.timestep_timer.hits)
        self.timestep_timer.reset()

    def write_out_performance(
        self,
        backend: str,
        is_orchestrated: bool,
        dt_atmos: float,
    ):
        if self.comm.Get_rank() == 0:
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
                git_hash = "None"
        else:
            git_hash = None
        git_hash = self.comm.bcast(git_hash, root=0)

        self.times_per_step.append(self.total_timer.times)
        self.hits_per_step.append(self.total_timer.hits)
        self.comm.Barrier()
        while {} in self.hits_per_step:
            self.hits_per_step.remove({})
        collect_data_and_write_to_file(
            len(self.hits_per_step) - 1,
            backend,
            is_orchestrated,
            git_hash,
            self.comm,
            self.hits_per_step,
            self.times_per_step,
            self.experiment_name,
            dt_atmos,
        )


class NullPerformanceCollector:
    def __init__(self):
        self.total_timer = pace.util.NullTimer()
        self.timestep_timer = pace.util.NullTimer()

    def collect_performance(self):
        pass

    def write_out_performance(
        self,
        backend: str,
        is_orchestrated: bool,
        dt_atmos: float,
    ):
        pass
