import collections
import functools
import hashlib
import inspect
import os
import types
from typing import BinaryIO, Callable, Sequence, Tuple, Union

import gt4py
import gt4py as gt
import numpy as np
import xarray as xr
import yaml
from fv3gfs.util import Quantity
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils

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
    for spec in arg_specs:
        if spec.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {spec.arg_name} is {spec.intent}, must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace_kwargs = {}
            for spec in arg_specs:
                arg_name, standard_name, units, intent = spec
                if standard_name not in state:
                    raise ValueError(f"{standard_name} not present in state")
                elif units != state[standard_name].units:
                    raise ValueError(
                        f"{standard_name} has units {state[standard_name].units} when {units} is required"
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


class FV3StencilObject:
    """GT4Py stencil object used for fv3core."""

    def __init__(self, stencil_object: gt4py.StencilObject, build_info: dict):
        self.stencil_object = stencil_object
        self._build_info = build_info

    @property
    def build_info(self) -> dict:
        """Return the build_info created when compiling the stencil."""
        return self._build_info

    def __call__(self, *args, **kwargs):
        return self.stencil_object(*args, **kwargs)


def _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs):
    flag_errmsg = (
        "The {} flag should be set in "
        + __name__
        + " instead of as an argument to stencil"
    )
    for flag in ("rebuild", "backend"):
        if flag in stencil_kwargs:
            raise ValueError(flag_errmsg.format(flag))


def gtstencil(definition=None, **stencil_kwargs) -> Callable[..., None]:
    _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs)

    def decorator(func) -> Callable[..., None]:
        stencils = {}
        times_called = 0

        def get_origin(func, call_args, call_kwargs):
            sig = inspect.signature(func)
            first_name = next(iter(sig.parameters))

            if len(call_args) == 0:
                first_storage = call_kwargs[first_name]
            else:
                first_storage = call_args[0]

            if not isinstance(first_storage, gt4py.storage.storage.Storage):
                raise TypeError(
                    "First stencil argument should be a gt4py.storage.storage.Storage."
                )

            origin: Sequence[int, ...]
            origin_arg = call_kwargs.get("origin", None)
            if isinstance(origin_arg, collections.Mapping):
                if first_name in origin_arg:
                    origin = origin_arg[first_name]
                else:
                    origin = first_storage.default_origin
            elif isinstance(origin_arg, collections.Sequence):
                origin = origin_arg
            elif origin_arg is None:
                origin = first_storage.default_origin
            else:
                raise TypeError(
                    f"Type of the origin argument was {type(origin_arg)}. Expected Sequence or Mapping."
                )

            return origin

        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> None:
            nonlocal times_called
            # This uses the module-level globals backend and rebuild (defined above)
            key = (global_config.get_backend(), global_config.get_rebuild())
            if key not in stencils:
                # Add globals to stencil_kwargs
                stencil_kwargs["rebuild"] = global_config.get_rebuild()
                stencil_kwargs["backend"] = global_config.get_backend()

                stencil_kwargs["externals"] = {
                    "namelist": spec.namelist,
                    "grid": spec.grid,
                    **stencil_kwargs.get("externals", dict()),
                }

                axis_offsets = spec.grid.axis_offsets(
                    origin=get_origin(func, args, kwargs)
                )
                stencil_kwargs["externals"].update(axis_offsets)

                # Generate stencil
                build_info = {}
                stencil = gtscript.stencil(build_info=build_info, **stencil_kwargs)(
                    func
                )
                stencils[key] = FV3StencilObject(stencil, build_info)
            name = f"{func.__module__}.{func.__name__}"
            _maybe_save_report(
                f"{name}-before",
                times_called,
                func.__dict__["_gtscript_"]["api_signature"],
                args,
                kwargs,
            )
            kwargs["validate_args"] = kwargs.get("validate_args", utils.validate_args)
            result = stencils[key](*args, **kwargs)
            _maybe_save_report(
                f"{name}-after",
                times_called,
                func.__dict__["_gtscript_"]["api_signature"],
                args,
                kwargs,
            )
            times_called += 1
            return result

        return wrapped

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
        if isinstance(arg, gt.storage.storage.Storage):
            args[i] = np.asarray(arg)
    for i, (name, value) in enumerate(kwargs_list):
        if isinstance(value, gt.storage.storage.Storage):
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
    if isinstance(arg, gt.storage.storage.Storage):
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
