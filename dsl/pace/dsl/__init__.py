import gt4py.config

from pace.util.mpi import MPI

from .stencil import FrozenStencil, StencilConfig, StencilFactory


if MPI is not None:
    gt4py.config.cache_settings["root_path"] = "."
    gt4py.config.cache_settings["dir_name"] = f".gt_cache_{MPI.COMM_WORLD.Get_rank()}"
