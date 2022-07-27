import copy
import dataclasses
import enum
import hashlib
import inspect
import re
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import dace
import gt4py
import numpy as np
from gt4py import gtscript
from gt4py.storage.storage import Storage
from gtc.passes.oir_pipeline import DefaultPipeline, OirPipeline

import pace.dsl.gt4py_utils as gt4py_utils
import pace.util
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.dace.orchestration import SDFGConvertible
from pace.dsl.gt4py_utils import is_gpu_backend
from pace.dsl.typing import Index3D, cast_to_index3d
from pace.util import testing
from pace.util.communicator import CubedSphereCommunicator
from pace.util.decomposition import (
    block_waiting_for_compilation,
    compiling_equivalent,
    determine_rank_is_compiling,
    set_distributed_caches,
    unblock_waiting_tiles,
)
from pace.util.halo_data_transformer import QuantityHaloSpec
from pace.util.mpi import MPI


class RunMode(enum.Enum):
    """
    Run-Mode for the model
        Build: compile & save compiled files only
        BuildAndRun: compile & save compiled files, then run
        Run: load from .so and run, will fail if .so is not available
    """

    Build = 0
    BuildAndRun = 1
    Run = 2


class CompilationConfig:
    def __init__(
        self,
        backend: str = "numpy",
        rebuild: bool = True,
        validate_args: bool = True,
        format_source: bool = False,
        device_sync: bool = False,
        run_mode: RunMode = RunMode.BuildAndRun,
        use_minimal_caching: bool = False,
        communicator: Optional[CubedSphereCommunicator] = None,
    ) -> None:
        if run_mode != RunMode.Run and use_minimal_caching:
            raise RuntimeError("Cannot use minimal caching in any building mode")
        if (not ("gpu" in backend or "cuda" in backend)) and device_sync is True:
            raise RuntimeError("Device sync is true on a CPU based backend")
        # GT4Py backend args
        self.backend = backend
        self.rebuild = rebuild
        self.validate_args = validate_args
        self.format_source = format_source
        self.device_sync = device_sync
        # Caching strategy
        self.run_mode = run_mode
        self.use_minimal_caching = use_minimal_caching
        (
            self.rank,
            self.size,
            self.compiling_equivalent,
            self.is_compiling,
        ) = self.get_decomposition_info_from_comm(communicator)
        if communicator:
            set_distributed_caches(self)

    def get_decomposition_info_from_comm(
        self, communicator: Optional[CubedSphereCommunicator]
    ) -> Tuple[int, int, int, bool]:
        if communicator:
            rank = communicator.rank
            size = communicator.partitioner.total_ranks
            if self.use_minimal_caching:
                equivalent_compiling_rank = compiling_equivalent(
                    rank, communicator.partitioner
                )
                is_compiling = determine_rank_is_compiling(
                    rank, communicator.partitioner
                )
            else:
                equivalent_compiling_rank = rank
                is_compiling = True
        else:
            rank = 1
            size = rank
            equivalent_compiling_rank = rank
            is_compiling = True

        return rank, size, equivalent_compiling_rank, is_compiling

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "rebuild": self.rebuild,
            "validate_args": self.validate_args,
            "format_source": self.format_source,
            "device_sync": self.device_sync,
            "run_mode": str(self.run_mode.name),
            "use_minimal_caching": self.use_minimal_caching,
        }

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls(
            backend=data.get("backend", "numpy"),
            rebuild=data.get("rebuild", False),
            validate_args=data.get("validate_args", True),
            format_source=data.get("format_source", False),
            device_sync=data.get("device_sync", False),
            run_mode=RunMode[data.get("run_mode", "BuildAndRun")],
            use_minimal_caching=data.get("use_minimal_caching", False),
            communicator=None,
        )
        return instance


@dataclasses.dataclass
class StencilConfig(Hashable):
    compare_to_numpy: bool = False
    compilation_config: CompilationConfig = CompilationConfig()
    dace_config: Optional[DaceConfig] = None

    def __post_init__(self):
        self.backend_opts = self._get_backend_opts(
            self.compilation_config.device_sync, self.compilation_config.format_source
        )
        self._hash = self._compute_hash()
        # We need a DaceConfig to known our orchestration as part of the build system
        # but we can't hash it very well (for now). The workaround is to make
        # sure we have a default Python orchestrated config.
        if self.dace_config is None:
            self.dace_config = DaceConfig(
                communicator=None,
                backend=self.compilation_config.backend,
                orchestration=DaCeOrchestration.Python,
            )

    def _compute_hash(self):
        md5 = hashlib.md5()
        md5.update(self.compilation_config.backend.encode())
        for attr in (
            self.compilation_config.rebuild,
            self.compilation_config.validate_args,
            self.compilation_config.use_minimal_caching,
            self.compare_to_numpy,
            self.backend_opts["format_source"],
        ):
            md5.update(bytes(attr))
        attr = self.backend_opts.get("device_sync", None)
        if attr:
            md5.update(bytes(attr))
        md5.update(bytes(self.compilation_config.run_mode.value))
        return int(md5.hexdigest(), base=16)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        try:
            return self.__hash__() == other.__hash__()
        except AttributeError:
            return False

    def _get_backend_opts(
        self,
        device_sync: Optional[bool] = None,
        format_source: Optional[bool] = None,
    ) -> Dict[str, Any]:
        backend_opts: Dict[str, Any] = {}
        all_backend_opts: Optional[Dict[str, Any]] = {
            "device_sync": {
                "backend": r".*(gpu|cuda)$",
                "value": False,
            },
            "format_source": {
                "value": False,
            },
            "verbose": {"backend": r"(gt:|cuda)", "value": False},
        }
        for name, option in all_backend_opts.items():
            using_option_backend = re.match(
                option.get("backend", ""), self.compilation_config.backend
            )
            if "backend" not in option or using_option_backend:
                backend_opts[name] = option["value"]

        if device_sync is not None:
            backend_opts["device_sync"] = device_sync
        if format_source is not None:
            backend_opts["format_source"] = format_source

        return backend_opts

    def stencil_kwargs(
        self, *, func: Callable[..., None], skip_passes: Iterable[str] = ()
    ):
        kwargs = {
            "backend": self.compilation_config.backend,
            "rebuild": self.compilation_config.rebuild,
            "name": func.__module__ + "." + func.__name__,
            **self.backend_opts,
        }
        if not self.is_gpu_backend:
            kwargs.pop("device_sync", None)
        if skip_passes or kwargs.get("skip_passes", ()):
            kwargs["oir_pipeline"] = StencilConfig._get_oir_pipeline(
                list(kwargs.pop("skip_passes", ())) + list(skip_passes)  # type: ignore
            )
        return kwargs

    @property
    def is_gpu_backend(self) -> bool:
        return is_gpu_backend(self.compilation_config.backend)

    @classmethod
    def _get_oir_pipeline(cls, skip_passes: Sequence[str]) -> OirPipeline:
        """Creates a DefaultPipeline with skip_passes properly initialized."""
        step_map = {step.__name__: step for step in DefaultPipeline.all_steps()}
        skip_steps = [step_map[pass_name] for pass_name in skip_passes]
        return DefaultPipeline(skip=skip_steps)
