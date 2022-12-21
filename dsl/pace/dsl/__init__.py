import gt4py.cartesian.config

from pace.util.mpi import MPI

from . import dace
from .dace.dace_config import DaceConfig, DaCeOrchestration
from .dace.orchestration import orchestrate, orchestrate_function
from .stencil import (
    CompilationConfig,
    FrozenStencil,
    GridIndexing,
    StencilConfig,
    StencilFactory,
)


if MPI is not None:
    import os

    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

__version__ = "0.2.0"
