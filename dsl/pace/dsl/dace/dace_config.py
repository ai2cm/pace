import dataclasses
import enum
from typing import Optional

from pace.util.communicator import CubedSphereCommunicator


class DaCeOrchestration(enum.Enum):
    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


@dataclasses.dataclass
class DaceConfig:
    _backend: str = ""
    _orchestrate: DaCeOrchestration = DaCeOrchestration.Python
    _communicator: Optional[CubedSphereCommunicator] = None

    def __post_init__(self):
        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        import os

        self.orchestrate = DaCeOrchestration[os.getenv("FV3_DACEMODE", "Python")]

    def init(self, communicator: CubedSphereCommunicator):
        self._communicator = communicator

    def is_dace_orchestrated(self) -> bool:
        if self._orchestrate and "dace" not in self._backend:
            raise RuntimeError(
                "DaceConfig: orchestration can only be leverage "
                f"on gtc:dace or gtc:dace:gpu not on {self._backend}"
            )
        return "dace" in self._backend and self.orchestrate

    def is_gpu_backend(self) -> bool:
        return "gpu" in self._backend

    def set_backend(self, backend: str) -> None:
        self._backend = backend

    def get_backend(self) -> str:
        return self._backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self._orchestrate

    def get_communicator(self) -> CubedSphereCommunicator:
        return self._communicator


dace_config = DaceConfig()
