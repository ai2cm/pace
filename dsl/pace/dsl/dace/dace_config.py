import dataclasses
from typing import Callable

from pace.util.global_config import getenv_bool


@dataclasses.dataclass
class DaceConfig:
    backend: str = ""
    orchestrate: bool = False

    def __post_init__(self):
        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        self.orchestrate = getenv_bool("FV3_DACEMODE", "False")

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

    def get_orchestrate(self) -> bool:
        return self.orchestrate


dace_config = DaceConfig()
