import dataclasses

import pace.util
from pace.util import NullProfiler, Profiler

from .collector import (
    AbstractPerformanceCollector,
    NullPerformanceCollector,
    PerformanceCollector,
)


@dataclasses.dataclass
class PerformanceConfig:
    """Performance stats collector.

    collect_performance: overall flag turning collection on/pff
    collect_cProfile: use cProfile for CPU Python profiling
    collect_communication: collect halo exchange details
    experiment_name: to be printed in the JSON summary
    json_all_rank_threshold: number of nodes above the full performance
        report for all nodes won't be written (rank 0 is always written)
    """

    collect_performance: bool = False
    collect_cProfile: bool = False
    collect_communication: bool = False
    experiment_name: str = "test"
    json_all_rank_threshold: int = 1000

    def build(self, comm: pace.util.Comm) -> AbstractPerformanceCollector:
        if self.collect_performance:
            return PerformanceCollector(experiment_name=self.experiment_name, comm=comm)
        else:
            return NullPerformanceCollector()

    def build_profiler(self):
        if self.collect_cProfile:
            return Profiler()
        else:
            return NullProfiler()
