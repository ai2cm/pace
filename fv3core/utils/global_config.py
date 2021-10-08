import functools
import hashlib
import os
import re
from collections.abc import Hashable
from importlib import resources
from typing import Any, Dict, Optional

import yaml


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


def read_backend_options_file():
    file = resources.open_binary("fv3core", "gt4py_options.yml")
    if file:
        options = yaml.safe_load(file)
        file.close()
        return options
    raise FileNotFoundError("config file 'fv3core/gt4py_options.yml' not found")


class StencilConfig(Hashable):
    _all_backend_opts: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        backend: str,
        rebuild: bool,
        validate_args: bool,
        format_source: Optional[bool] = None,
        device_sync: Optional[bool] = None,
    ):
        self.backend = backend
        self.rebuild = rebuild
        self.validate_args = validate_args
        self.backend_opts = self._get_backend_opts(device_sync, format_source)
        self._hash = self._compute_hash()

    def _compute_hash(self):
        md5 = hashlib.md5()
        md5.update(self.backend.encode())
        for attr in (
            self.rebuild,
            self.validate_args,
            self.backend_opts["format_source"],
        ):
            md5.update(bytes(attr))
        attr = self.backend_opts.get("device_sync", None)
        if attr:
            md5.update(bytes(attr))
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
        if not StencilConfig._all_backend_opts:
            StencilConfig._all_backend_opts = read_backend_options_file()
        all_backend_opts: Dict[str, Any] = StencilConfig._all_backend_opts

        backend_opts: Dict[str, Any] = {}
        for name, option in all_backend_opts.items():
            using_option_backend = re.match(option.get("backend", ""), get_backend())
            if "backend" not in option or using_option_backend:
                backend_opts[name] = option["value"]

        if device_sync is not None:
            backend_opts["device_sync"] = device_sync
        if format_source is not None:
            backend_opts["format_source"] = format_source

        return backend_opts

    @property
    def stencil_kwargs(self):
        kwargs = {
            "backend": self.backend,
            "rebuild": self.rebuild,
            **self.backend_opts,
        }
        if not is_gpu_backend():
            kwargs.pop("device_sync", None)
        return kwargs


def get_stencil_config():
    return StencilConfig(
        backend=get_backend(),
        rebuild=get_rebuild(),
        validate_args=get_validate_args(),
    )


# Options: numpy, gtx86, gtcuda, debug
_BACKEND: Optional[str] = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD: bool = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_VALIDATE_ARGS: bool = True
