import enum
from typing import Optional

import dace.config

from pace.util.communicator import CubedSphereCommunicator


class DaCeOrchestration(enum.Enum):
    """
    Orchestration mode for DaCe

        Python: python orchestration
        Build: compile & save SDFG only
        BuildAndRun: compile & save SDFG, then run
        Run: load from .so and run, will fail if .so is not available
    """

    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


class DaceConfig:
    def __init__(
        self,
        communicator: Optional[CubedSphereCommunicator],
        backend: str,
        orchestration: Optional[DaCeOrchestration] = None,
    ):
        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        import os

        if orchestration is None:
            self._orchestrate = DaCeOrchestration[os.getenv("FV3_DACEMODE", "Python")]
        else:
            self._orchestrate = orchestration

        # Set the configuration of DaCe to a rigid & tested set of divergence
        # from the defaults when orchestrating
        if orchestration != DaCeOrchestration.Python:
            # Required to True for gt4py storage/memory
            dace.config.Config.set(
                "compiler",
                "allow_view_arguments",
                value=True,
            )
            # Removed --fmath
            dace.config.Config.set(
                "compiler",
                "cpu",
                "args",
                value="-std=c++14 -fPIC -Wall -Wextra -O3",
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cpu",
                "openmp_sections",
                value=0,
            )
            # Removed --fast-math
            dace.config.Config.set(
                "compiler",
                "cuda",
                "args",
                value="-std=c++14 -Xcompiler -fPIC -O3 -Xcompiler -march=native",
            )
            # Potentiall buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cuda",
                "max_concurrent_streams",
                value=-1,
            )
            # Speed up built time
            dace.config.Config.set(
                "compiler",
                "cuda",
                "unique_functions",
                value="none",
            )
            # Required to False. Bug to be fixes on DaCe side
            dace.config.Config.set(
                "execution",
                "general",
                "check_args",
                value=False,
            )
            # Required for HaloEx callbacks and general code sanity
            dace.config.Config.set(
                "frontend",
                "dont_fuse_callbacks",
                value=True,
            )
            # Unroll all loop - outer loop should be exempted with dace.nounroll
            dace.config.Config.set(
                "frontend",
                "unroll_threshold",
                value=False,
            )
            # Allow for a longer stack dump when parsing fails
            dace.config.Config.set(
                "frontend",
                "verbose_errors",
                value=True,
            )
            # Build speed up by removing some deep copies
            dace.config.Config.set(
                "store_history",
                value=False,
            )

        # attempt to kill the dace.conf to avoid confusion
        if dace.config.Config._cfg_filename:
            try:
                import os

                os.remove(dace.config.Config._cfg_filename)
            except OSError:
                pass

        self._backend = backend
        from pace.dsl.dace.build import (
            read_target_rank,
            set_distributed_caches,
            write_decomposition,
        )

        if (
            communicator
            and (
                self._orchestrate == DaCeOrchestration.Build
                or self._orchestrate == DaCeOrchestration.BuildAndRun
            )
            and communicator.rank == 0
            and communicator.comm.Get_size() > 1
        ):
            write_decomposition(communicator.partitioner)

        # Distributed build required info
        if communicator:
            self.my_rank = communicator.rank
            self.rank_size = communicator.comm.Get_size()
            from gt4py import config as gt_config

            config_path = (
                f"{gt_config.cache_settings['root_path']}/.layout/decomposition.yml"
            )
            self.target_rank = read_target_rank(
                rank=self.my_rank,
                partitioner=communicator.partitioner,
                config=self,
                layout_filepath=config_path,
            )
        else:
            self.my_rank = 0
            self.rank_size = 1
            self.target_rank = 0

        set_distributed_caches(self)

        if (
            self._orchestrate != DaCeOrchestration.Python
            and "dace" not in self._backend
        ):
            raise RuntimeError(
                "DaceConfig: orchestration can only be leverage "
                f"on gtc:dace or gtc:dace:gpu not on {self._backend}"
            )

    def is_dace_orchestrated(self) -> bool:
        return self._orchestrate != DaCeOrchestration.Python

    def is_gpu_backend(self) -> bool:
        return "gpu" in self._backend

    def get_backend(self) -> str:
        return self._backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self._orchestrate
