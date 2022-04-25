import dataclasses
from pace.util import communicator

from pace.util.global_config import getenv_bool
from pace.util.communicator import CubedSphereCommunicator, CubedSpherePartitioner
from typing import Optional
import enum


class DaCeOrchestration(enum.Enum):
    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


@dataclasses.dataclass
class DaceConfig:
    backend: str = ""
    orchestrate: DaCeOrchestration = DaCeOrchestration.Python
    communicator: Optional[CubedSphereCommunicator] = None

    def __post_init__(self):
        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        self.orchestrate = (
            DaCeOrchestration.BuildAndRun
            if getenv_bool("FV3_DACEMODE", "False")
            else DaCeOrchestration.Python
        )

    def init(self, communicator: CubedSphereCommunicator):
        self.communicator = communicator

    def is_dace_orchestrated(self) -> bool:
        if self.orchestrate and "dace" not in self.backend:
            raise RuntimeError(
                "DaceConfig: orchestration can only be leverage "
                f"on gtc:dace or gtc:dace:gpu not on {self.backend}"
            )
        return "dace" in self.backend and self.orchestrate

    def is_gpu_backend(self) -> bool:
        return "gpu" in self.backend

    def get_backend(self) -> str:
        return self.backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self.orchestrate

    def get_communicator(self) -> CubedSphereCommunicator:
        if not self.communicator:
            raise RuntimeError(
                "DaceConfig: orchestration didn't specify the communicator"
            )
        return self.communicator


dace_config = DaceConfig()
