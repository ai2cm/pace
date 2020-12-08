import ast
import os


def getenv_bool(name: str, default: str) -> bool:
    indicator = os.getenv(name, default).title()
    return indicator == "True"


def set_backend(new_backend: str):
    global _BACKEND
    _BACKEND = new_backend


def get_backend() -> str:
    return _BACKEND


def set_rebuild(flag: bool):
    global _REBUILD
    _REBUILD = flag


def get_rebuild() -> bool:
    return _REBUILD


_BACKEND = None  # Options: numpy, gtx86, gtcuda, debug
_REBUILD = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "True")
