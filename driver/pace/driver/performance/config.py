import dataclasses

import pace.util

from .collector import (
    AbstractPerformanceCollector,
    NullPerformanceCollector,
    PerformanceCollector,
)


@dataclasses.dataclass
class PerformanceConfig:
    collect_performance: bool = False
    experiment_name: str = "test"

    def build(self, comm: pace.util.Comm) -> AbstractPerformanceCollector:
        if self.collect_performance:
            return PerformanceCollector(experiment_name=self.experiment_name, comm=comm)
        else:
            return NullPerformanceCollector()
