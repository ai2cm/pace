import enum
from typing import Any, Dict, Optional

import dace.config

from pace.dsl.gt4py_utils import is_gpu_backend
from pace.util.communicator import CubedSphereCommunicator


# TODO (floriand): Temporary deactivate the distributed compiled
# until we deal with the Grid data inlining during orchestration
# See github issue #301
DEACTIVATE_DISTRIBUTED_DACE_COMPILE = True


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
        tile_nx: int = 0,
        tile_nz: int = 0,
        orchestration: Optional[DaCeOrchestration] = None,
    ):
        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        import os

        if orchestration is None:
            fv3_dacemode_env_var = os.getenv("FV3_DACEMODE", "Python")
            # The below condition guard against defining empty FV3_DACEMODE and
            # awkward behavior of os.getenv returning "" even when not defined
            if fv3_dacemode_env_var is None or fv3_dacemode_env_var == "":
                fv3_dacemode_env_var = "Python"
            self._orchestrate = DaCeOrchestration[fv3_dacemode_env_var]
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
            dace.config.Config.set("compiler", "cuda", "cuda_arch", value="60")
            dace.config.Config.set(
                "compiler", "cuda", "default_block_size", value="64,8,1"
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cuda",
                "max_concurrent_streams",
                value=-1,  # no concurrent streams, every kernel on defaultStream
            )
            # Speed up built time
            dace.config.Config.set(
                "compiler",
                "cuda",
                "unique_functions",
                value="none",
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

            # Enable to debug GPU failures
            dace.config.Config.set("compiler", "cuda", "syncdebug", value=False)

        # attempt to kill the dace.conf to avoid confusion
        if dace.config.Config._cfg_filename:
            try:
                import os

                os.remove(dace.config.Config._cfg_filename)
            except OSError:
                pass

        self._backend = backend
        self.tile_resolution = [tile_nx, tile_nx, tile_nz]
        from pace.dsl.dace.build import get_target_rank, set_distributed_caches

        # Distributed build required info
        if communicator:
            self.my_rank = communicator.rank
            self.rank_size = communicator.comm.Get_size()
            if DEACTIVATE_DISTRIBUTED_DACE_COMPILE:
                self.target_rank = communicator.rank
            else:
                self.target_rank = get_target_rank(
                    self.my_rank, communicator.partitioner
                )
            self.layout = communicator.partitioner.layout
        else:
            self.my_rank = 0
            self.rank_size = 1
            self.target_rank = 0
            self.layout = [1, 1]

        set_distributed_caches(self)

        if (
            self._orchestrate != DaCeOrchestration.Python
            and "dace" not in self._backend
        ):
            raise RuntimeError(
                "DaceConfig: orchestration can only be leverage "
                f"on dace or dace:gpu not on {self._backend}"
            )

    def is_dace_orchestrated(self) -> bool:
        return self._orchestrate != DaCeOrchestration.Python

    def is_gpu_backend(self) -> bool:
        return is_gpu_backend(self._backend)

    def get_backend(self) -> str:
        return self._backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self._orchestrate

    def get_sync_debug(self) -> bool:
        return dace.config.Config.get("compiler", "cuda", "syncdebug")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "_orchestrate": str(self._orchestrate.name),
            "_backend": self._backend,
            "my_rank": self.my_rank,
            "rank_size": self.rank_size,
            "layout": self.layout,
            "tile_resolution": self.tile_resolution,
        }

    @classmethod
    def from_dict(cls, data: dict):
        config = cls(
            None,
            backend=data["_backend"],
            orchestration=DaCeOrchestration[data["_orchestrate"]],
        )
        config.my_rank = data["my_rank"]
        config.rank_size = data["rank_size"]
        config.layout = data["layout"]
        config.tile_resolution = data["tile_resolution"]
        return config
