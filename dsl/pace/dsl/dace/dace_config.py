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


# See description below. Dangerous, should be refactored out
# Either we can JIT properly via GTC or the compute path need to be able
# to trigger compilation at call time properly if you haven't anyone above you
def is_dacemode_codegen_whitelisted(func: Callable[..., None]) -> bool:
    """Whitelist of stencil function that need code generation in DACE mode.
    Some stencils are called within the __init__ and therefore will need to
    be pre-compiled nonetheless.
    """
    whitelist = [
        "dp_ref_compute",
        "cubic_spline_interpolation_constants",
        "calc_damp",
        "set_gz",
        "set_pem",
        "copy_defn",
        "compute_geopotential",
        # DynamicalCore
        "init_pfull",
        # CubedToLatLon for Metric/Grid/baroclinic state calculation
        "ord4_transform",
        "c2l_ord2",
        # Expanded grid variable
        "compute_coriolis_parameter_defn",
    ]
    return any(func.__name__ in name for name in whitelist)


dace_config = DaceConfig()
