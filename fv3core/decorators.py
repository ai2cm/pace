import collections
import functools
import hashlib
import os
import pickle
import types
from typing import Any, BinaryIO, Callable, Dict, Optional

import gt4py
import gt4py.storage as gt_storage
import numpy as np
import yaml
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import Index3D

from .utils import global_config


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def enable_stencil_report(
    *, path: str, save_args: bool, save_report: bool, include_halos: bool = False
):
    global stencil_report_path
    global save_stencil_args
    global save_stencil_report
    global report_include_halos
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    stencil_report_path = path
    save_stencil_args = save_args
    save_stencil_report = save_report
    report_include_halos = include_halos


def disable_stencil_report():
    global stencil_report_path
    global save_stencil_args
    global save_stencil_report
    stencil_report_path = None
    save_stencil_args = False
    save_stencil_report = False


stencil_report_path = None
save_stencil_args = False
save_stencil_report = False
report_include_halos = False


def state_inputs(*arg_specs):
    for sp in arg_specs:
        if sp.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {sp.arg_name} is {sp.intent}, "
                "must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace_kwargs = {}
            for sp in arg_specs:
                arg_name, standard_name, units, intent = sp
                if standard_name not in state:
                    raise ValueError(f"{standard_name} not present in state")
                elif units != state[standard_name].units:
                    raise ValueError(
                        f"{standard_name} has units "
                        f"{state[standard_name].units} when {units} is required"
                    )
                elif intent not in VALID_INTENTS:
                    raise ValueError(
                        f"expected intent to be one of {VALID_INTENTS}, got {intent}"
                    )
                else:
                    namespace_kwargs[arg_name] = state[standard_name].storage
                    namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
            func(types.SimpleNamespace(**namespace_kwargs), *args, **kwargs)

        return wrapped

    return decorator


def _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs):
    flag_errmsg = (
        "The {} flag should be set in fv3core.utils.global_config.py"
        "instead of as an argument to stencil"
    )
    for flag in ("rebuild", "backend"):
        if flag in stencil_kwargs:
            raise ValueError(flag_errmsg.format(flag))


class StencilDataCache(collections.abc.Mapping):
    """
    A Python object cache along with stencils.

    This uses both the disk and an in-memory map.
    """

    def __init__(self, extension: str = "cache.py"):
        self.extension: str = extension
        """Extension used for filenames in cache."""

        self.cache: Dict[str, Any] = {}
        """In-memory cache of the data pickled to disk."""

    def _get_cache_filename(self, stencil: gt4py.StencilObject) -> str:
        pymodule_filename = stencil._file_name
        return f"{os.path.splitext(pymodule_filename)[0]}_{self.extension}"

    def __getitem__(self, stencil: gt4py.StencilObject):
        key = hash(stencil)
        if key not in self.cache:
            filename = self._get_cache_filename(stencil)
            if os.path.exists(filename):
                self.cache[key] = pickle.load(open(filename, mode="rb"))
        return self.cache[key] if key in self.cache else {}

    def __setitem__(self, stencil: gt4py.StencilObject, value: Any):
        key = hash(stencil)
        filename = self._get_cache_filename(stencil)
        self.cache[key] = value
        pickle.dump(self.cache[key], open(filename, mode="wb"))
        return self.cache[key]

    def __len__(self) -> int:
        return len(self.cache)

    def __iter__(self):
        return self.cache.__iter__()


class FV3StencilObject:
    """GT4Py stencil object used for fv3core."""

    def __init__(self, func: Callable[..., None], **kwargs):
        self.func: Callable[..., None] = func
        """The definition function."""

        self.stencil_object: Optional[gt4py.StencilObject] = None
        """The current generated stencil object returned from gt4py."""

        self.times_called: int = 0
        """Number of times this stencil has been called."""

        self.timers: Dict[str, float] = types.SimpleNamespace(call_run=0.0, run=0.0)
        """Accumulated time spent in this stencil.

        call_run includes stencil call overhead, while run omits it."""

        self._passed_externals: Dict[str, Any] = kwargs.pop("externals", {})
        """Externals passed in the decorator (others are added later)."""

        self.backend_kwargs: Dict[str, Any] = kwargs
        """Remainder of the arguments assumed to be compiler backend options."""

        self._data_cache: StencilDataCache = StencilDataCache("data_cache.p")
        """Data cache to store axis offsets and passed externals."""

    @property
    def built(self) -> bool:
        """Indicates whether the stencil is loaded."""
        return self.stencil_object is not None

    @property
    def axis_offsets(self) -> Dict[str, Any]:
        """AxisOffsets used in this stencil."""
        cached_data = self._data_cache[self.stencil_object]
        return cached_data["axis_offsets"] if "axis_offsets" in cached_data else {}

    @property
    def passed_externals(self) -> Dict[str, Any]:
        """Passed externals used in this stencil."""
        cached_data = self._data_cache[self.stencil_object]
        return (
            cached_data["passed_externals"] if "passed_externals" in cached_data else {}
        )

    def _check_axis_offsets(self, axis_offsets: Dict[str, Any]) -> bool:
        for key, value in self.axis_offsets.items():
            if axis_offsets[key] != value:
                return True
        return False

    def _check_passed_externals(self) -> bool:
        passed_externals = self.passed_externals
        for key, value in self._passed_externals.items():
            if passed_externals[key] != value:
                return True
        return False

    def __call__(self, *args, origin: Index3D, domain: Index3D, **kwargs) -> None:
        """Call the stencil, compiling the stencil if necessary.

        The stencil needs to be recompiled if any of the following changes
        1. the origin and/or domain
        2. any external value
        3. the function signature or code

        Args:
            domain: Stencil compute domain (required)
            origin: Data index mapped to (0, 0, 0) in the compute domain (required)
        """

        # Can optimize this by marking stencils that need these
        axis_offsets = fv3core.utils.axis_offsets(spec.grid, origin, domain)

        regenerate_stencil = not self.built or global_config.get_rebuild()

        # Check if we really do need to regenerate
        if not regenerate_stencil:
            axis_offsets_changed = self._check_axis_offsets(axis_offsets)
            regenerate_stencil = regenerate_stencil or axis_offsets_changed

        if self._passed_externals and not regenerate_stencil:
            passed_externals_changed = self._check_passed_externals()
            regenerate_stencil = regenerate_stencil or passed_externals_changed

        if regenerate_stencil:
            new_build_info = {}
            stencil_kwargs = {
                "rebuild": global_config.get_rebuild(),
                "backend": global_config.get_backend(),
                "externals": {
                    "namelist": spec.namelist,
                    "grid": spec.grid,
                    **axis_offsets,
                    **self._passed_externals,
                },
                "format_source": global_config.get_format_source(),
                "build_info": new_build_info,
                **self.backend_kwargs,
            }

            # gtscript.stencil always returns a new class instance even if it
            # used the cached module.
            self.stencil_object = gtscript.stencil(
                definition=self.func, **stencil_kwargs
            )
            stencil = self.stencil_object
            if (
                stencil not in self._data_cache
                and "def_ir" in stencil_kwargs["build_info"]
            ):
                def_ir = stencil_kwargs["build_info"]["def_ir"]
                axis_offsets = {
                    k: v for k, v in def_ir.externals.items() if k in axis_offsets
                }
                self._data_cache[stencil] = dict(
                    axis_offsets=axis_offsets, passed_externals=self._passed_externals
                )

        # Call it
        exec_info = {}
        kwargs["exec_info"] = kwargs.get("exec_info", exec_info)
        kwargs["validate_args"] = kwargs.get("validate_args", utils.validate_args)
        name = f"{self.func.__module__}.{self.func.__name__}"
        _maybe_save_report(
            f"{name}-before",
            self.times_called,
            self.func.__dict__["_gtscript_"]["api_signature"],
            args,
            kwargs,
        )
        self.stencil_object(*args, **kwargs, origin=origin, domain=domain)
        _maybe_save_report(
            f"{name}-after",
            self.times_called,
            self.func.__dict__["_gtscript_"]["api_signature"],
            args,
            kwargs,
        )
        self.times_called += 1

        # Update timers
        exec_info = kwargs["exec_info"]
        self.timers.run += exec_info["run_end_time"] - exec_info["run_start_time"]
        self.timers.call_run += (
            exec_info["call_run_end_time"] - exec_info["call_run_start_time"]
        )


def gtstencil(definition=None, **stencil_kwargs) -> Callable[..., None]:
    _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs)

    def decorator(func) -> FV3StencilObject:
        return FV3StencilObject(func, **stencil_kwargs)

    if definition is None:
        return decorator
    else:
        return decorator(definition)


def _get_case_name(name, times_called):
    return f"stencil-{name}-n{times_called:04d}"


def _get_report_filename():
    return f"stencil-report-r{spec.grid.rank:03d}.yml"


def _maybe_save_report(name, times_called, arg_infos, args, kwargs):
    case_name = _get_case_name(name, times_called)
    if save_stencil_args:
        args_filename = os.path.join(stencil_report_path, f"{case_name}.npz")
        with open(args_filename, "wb") as f:
            _save_args(f, args, kwargs)
    if save_stencil_report:
        report_filename = os.path.join(stencil_report_path, _get_report_filename())
        with open(report_filename, "a") as f:
            yaml.safe_dump({case_name: _get_stencil_report(arg_infos, args, kwargs)}, f)


def _save_args(file: BinaryIO, args, kwargs):
    args = list(args)
    kwargs_list = sorted(list(kwargs.items()))
    for i, arg in enumerate(args):
        if isinstance(arg, gt_storage.Storage):
            args[i] = np.asarray(arg)
    for i, (name, value) in enumerate(kwargs_list):
        if isinstance(value, gt_storage.Storage):
            kwargs_list[i] = (name, np.asarray(value))
    np.savez(file, *args, **dict(kwargs_list))


def _get_stencil_report(arg_infos, args, kwargs):
    return {
        "args": _get_args_report(arg_infos, args),
        "kwargs": _get_kwargs_report(kwargs),
    }


def _get_args_report(arg_infos, args):
    report = {}
    for argi in range(len(args)):
        report[arg_infos[argi].name] = _get_arg_report(args[argi])
    return report


def _get_kwargs_report(kwargs):
    return {name: _get_arg_report(value) for (name, value) in kwargs.items()}


def _get_arg_report(arg):
    if isinstance(arg, gt_storage.storage.Storage):
        arg = np.asarray(arg)
    if isinstance(arg, np.ndarray):
        if not report_include_halos:
            islice = slice(spec.grid.is_, spec.grid.ie + 1)
            jslice = slice(spec.grid.js, spec.grid.je + 1)
            arg = arg[islice, jslice, :]
        return {
            "md5": hashlib.md5(arg.tobytes()).hexdigest(),
            "min": float(arg.min()),
            "max": float(arg.max()),
            "mean": float(arg.mean()),
            "std": float(arg.std()),
        }
    else:
        return str(arg)
