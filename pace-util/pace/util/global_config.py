import functools
import os
from typing import Optional
import pace.util as util

def getenv_bool(name: str, default: str) -> bool:
    indicator = os.getenv(name, default).title()
    return indicator == "True"


def set_backend(new_backend: str):
    global _BACKEND
    _BACKEND = new_backend
    for function in (is_gpu_backend, is_gtc_backend):
        function.cache_clear()


def get_backend() -> str:
    return _BACKEND


def set_rebuild(flag: bool):
    global _REBUILD
    _REBUILD = flag


def get_rebuild() -> bool:
    return _REBUILD


def set_validate_args(new_validate_args: bool):
    global _VALIDATE_ARGS
    _VALIDATE_ARGS = new_validate_args


# Set to "False" to skip validating gt4py stencil arguments
@functools.lru_cache(maxsize=None)
def get_validate_args() -> bool:
    return _VALIDATE_ARGS


@functools.lru_cache(maxsize=None)
def is_gpu_backend() -> bool:
    return get_backend().endswith("cuda") or get_backend().endswith("gpu")


@functools.lru_cache(maxsize=None)
def is_gtc_backend() -> bool:
    return get_backend().startswith("gtc")


# Options: numpy, gtx86, gtcuda, debug
_BACKEND: Optional[str] = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD: bool = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_VALIDATE_ARGS: bool = True

import enum


class DaCeOrchestration(enum.Enum):
    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


def load_dace_orchestration() -> DaCeOrchestration:
    return DaCeOrchestration[os.getenv("FV3_DACEMODE", "Python")]


def get_dacemode() -> DaCeOrchestration:
    global _DACEMODE
    return _DACEMODE


def is_dace_orchestrated() -> bool:
    return _DACEMODE != DaCeOrchestration.Python


def set_dacemode(dacemode: DaCeOrchestration):
    global _DACEMODE
    _DACEMODE = dacemode


# Python: python orchestration
# Build: compile & save SDFG only
# BuildAndRun: compile & save SDFG, then run
# Run: load from .so and run, will fail if .so is not available
_DACEMODE: DaCeOrchestration = load_dace_orchestration()

def get_partitioner() -> Optional[util.CubedSpherePartitioner]:
    global _PARTITIONER
    return _PARTITIONER


def set_partitioner(partitioner: Optional[util.CubedSpherePartitioner]) -> None:
    global _PARTITIONER
    if _PARTITIONER is not None:
        print("re-setting the partitioner, why is that?")
    _PARTITIONER = partitioner


def set_partitioner_once(partitioner: Optional[util.CubedSpherePartitioner]) -> None:
    global _PARTITIONER
    if _PARTITIONER is None:
        _PARTITIONER = partitioner


# Partitioner from pace
_PARTITIONER: Optional[util.CubedSpherePartitioner] = None
