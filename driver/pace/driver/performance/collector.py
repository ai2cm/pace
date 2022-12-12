import copy
import os.path
import subprocess
from collections.abc import Mapping
from typing import List, Protocol

import numpy as np

import pace.util
from pace.driver.performance.report import (
    Report,
    TimeReport,
    collect_keys_from_data,
    gather_hit_counts,
    get_experiment_info,
    write_to_timestamped_json,
)
from pace.util._optional_imports import cupy as cp
from pace.util.utils import GPU_AVAILABLE

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

    def write_out_rank_0(
        self, backend: str, is_orchestrated: bool, dt_atmos: float, sim_status: str
    ):
        ...

    @classmethod
    def start_cuda_profiler(cls):
        if GPU_AVAILABLE:
            cp.cuda.profiler.start()

    @classmethod
    def stop_cuda_profiler(cls):
        if GPU_AVAILABLE:
            cp.cuda.profiler.stop()

    @classmethod
    def mark_cuda_profiler(cls, message: str):
        if GPU_AVAILABLE:
            cp.cuda.nvtx.Mark(message)


class PerformanceCollector(AbstractPerformanceCollector):
    def __init__(self, experiment_name: str, comm: pace.util.Comm):
        self.times_per_step: List[Mapping[str, float]] = []
        self.hits_per_step: List[Mapping[str, int]] = []
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

    def write_out_rank_0(
        self, backend: str, is_orchestrated: bool, dt_atmos: float, sim_status: str
    ):
        if self.comm.Get_rank() == 0:
            git_hash = "None"
            while {} in self.hits_per_step:
                self.hits_per_step.remove({})
            keys = collect_keys_from_data(self.times_per_step)
            data: List[float] = []
            timing_info = {}
            for timer_name in keys:
                data.clear()
                for data_point in self.times_per_step:
                    if timer_name in data_point:
                        data.append(data_point[timer_name])
                timing_info[timer_name] = TimeReport(
                    hits=0, times=copy.deepcopy(np.array(data).tolist())
                )
            exp_info = get_experiment_info(
                self.experiment_name,
                len(self.hits_per_step) - 1,
                backend,
                git_hash,
                is_orchestrated,
            )
            timing_info = gather_hit_counts(self.hits_per_step, timing_info)
            report = Report(
                setup=exp_info,
                times=timing_info,
                dt_atmos=dt_atmos,
                sim_status=sim_status,
            )
            write_to_timestamped_json(report)
        else:
            pass

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


class NullPerformanceCollector(AbstractPerformanceCollector):
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

    def write_out_rank_0(
        self, backend: str, is_orchestrated: bool, dt_atmos: float, sim_status: str
    ):
        pass
