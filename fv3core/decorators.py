import collections
import collections.abc
import functools
import hashlib
import os
import pickle
import types
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

import gt4py
import gt4py.storage as gt_storage
import numpy as np
import yaml
from gt4py import gtscript
from gt4py.definitions import Shape

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.global_config as global_config
from fv3core.utils.typing import Index3D


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
            namespace = get_namespace(arg_specs, state)
            func(namespace, *args, **kwargs)

        return wrapped

    return decorator


def get_namespace(arg_specs, state):
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
    return types.SimpleNamespace(**namespace_kwargs)


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

        self.cache: Dict[int, Any] = {}
        """In-memory cache of the data pickled to disk."""

    def _get_cache_filename(self, stencil: gt4py.StencilObject) -> str:
        pymodule_filename = stencil._file_name
        return f"{os.path.splitext(pymodule_filename)[0]}_{self.extension}"

    def __getitem__(self, stencil: gt4py.StencilObject) -> Any:
        key = hash(stencil)
        if key not in self.cache:
            filename = self._get_cache_filename(stencil)
            if os.path.exists(filename):
                self.cache[key] = pickle.load(open(filename, mode="rb"))
        return self.cache[key] if key in self.cache else {}

    def __setitem__(self, stencil: gt4py.StencilObject, value: Any) -> None:
        key = hash(stencil)
        filename = self._get_cache_filename(stencil)
        self.cache[key] = value
        pickle.dump(self.cache[key], open(filename, mode="wb"))
        return self.cache[key]

    def __contains__(self, stencil: gt4py.StencilObject) -> bool:
        return self[stencil]

    def __len__(self) -> int:
        return len(self.cache)

    def __iter__(self):
        return self.cache.__iter__()


class StencilWrapper:
    """Wrapped GT4Py stencil object."""

    def __init__(
        self,
        func: Callable[..., None],
        origin: Optional[Tuple[int, ...]] = None,
        domain: Optional[Index3D] = None,
        *,
        disable_cache: bool = False,
        delay_compile: Optional[bool] = None,
        device_sync: Optional[bool] = None,
        **kwargs,
    ):
        self.func: Callable = func
        """The definition function."""

        self.is_cached: bool = False
        """Flag to check if the stencil's runtime information is fully cached."""

        self.origin: Tuple[int, ...] = origin
        """The compute origin."""

        if domain is not None:
            domain = Shape(domain)
        self.domain: Optional[Shape] = domain
        """The compute domain."""

        self.disable_cache: bool = disable_cache
        """Disable caching if true."""

        self.backend: str = global_config.get_backend()
        """The gt4py backend name."""

        self.rebuild: bool = global_config.get_rebuild()
        """The gt4py stencil is rebuilt if true."""

        self.validate_args: bool = global_config.get_validate_args()
        """The gt4py stencil validates the arguments upon call invocation if true."""

        self.format_source: bool = global_config.get_format_source()
        """The gt4py generated code is formatted if true."""

        self.device_sync: bool = device_sync
        """Synchronize device (GPU) after each stencil call if true."""

        self.field_origins: Dict[str, Tuple[int, ...]] = {}
        """A dictionary of data field origins."""

        stencil_object = None
        if not delay_compile:
            self._set_device_sync(kwargs)

            stencil_object = gtscript.stencil(
                definition=self.func,
                backend=self.backend,
                rebuild=self.rebuild,
                format_source=self.format_source,
                **kwargs,
            )

        self.stencil_object: Optional[gt4py.StencilObject] = stencil_object
        """The current generated stencil object returned from gt4py."""

        self.arg_names: List[str] = []
        """List of argument names."""

    def clear(self):
        """Clears cached data items."""
        self.is_cached = False
        self.field_origins.clear()
        self.arg_names.clear()

    def __call__(
        self,
        *args,
        origin: Optional[Tuple[int, ...]] = None,
        domain: Optional[Index3D] = None,
        **kwargs,
    ) -> None:
        if self.is_cached and not self.validate_args:
            kwargs = self._process_kwargs(domain, *args, **kwargs)
            self.stencil_object.run(**kwargs, exec_info=None)
        else:
            if self.origin:
                assert origin is None, "cannot override origin provided at init"
                origin = self.origin
            self.field_origins = self._compute_field_origins(origin, *args, **kwargs)

            if self.domain:
                assert domain is None, "cannot override domain provided at init"
                domain = self.domain
            else:
                assert domain is not None, "no domain provided at call time"

            self.validate_args = global_config.get_validate_args()

            if self.validate_args:
                self.stencil_object(
                    *args,
                    **kwargs,
                    origin=self.field_origins,
                    domain=domain,
                    validate_args=True,
                )
            else:
                kwargs = self._process_kwargs(domain, *args, **kwargs)
                self.stencil_object.run(**kwargs, exec_info=None)
            self.is_cached = True

    def _process_kwargs(self, domain: Optional[Index3D], *args, **kwargs):
        """Processes keyword args for direct calls to stencil_object.run."""

        if domain is None:
            domain = self.domain
        if not self.arg_names:
            self.arg_names = self.field_names + self.parameter_names

        kwargs.update({"_origin_": self.field_origins, "_domain_": Shape(domain)})
        kwargs.update({name: arg for name, arg in zip(self.arg_names, args)})
        return kwargs

    def _compute_field_origins(
        self, origin: Tuple[int, ...], *args, **kwargs
    ) -> Dict[str, Tuple[int, ...]]:
        """Computes the origin for each field in the stencil call."""

        field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
        field_names: List[str] = self.field_names
        for i in range(len(field_names)):
            field_name = field_names[i]
            field_axes = (
                self.stencil_object.field_info[field_name].axes
                if self.stencil_object.field_info[field_name]
                else []
            )
            field_shape = args[i].shape if i < len(args) else kwargs[field_name].shape
            if field_axes == ["K"]:
                field_origin = [min(field_shape[0] - 1, origin[2])]
            else:
                field_origin = [
                    min(field_shape[j] - 1, origin[j]) for j in range(len(field_shape))
                ]
            field_origins[field_name] = tuple(field_origin)
        return field_origins

    def _set_device_sync(self, stencil_kwargs: Dict[str, Any]):
        """Sets the 'device_sync' backend option in the stencil kwargs."""
        if self.device_sync is None and "cuda" in self.backend:
            self.device_sync = global_config.get_device_sync()
        if self.device_sync is not None:
            stencil_kwargs["device_sync"] = self.device_sync

    @property
    def field_names(self) -> List[str]:
        """Returns the list of stencil field names."""
        return list(self.stencil_object.field_info.keys())

    @property
    def parameter_names(self) -> List[str]:
        """Returns the list of stencil parameter names."""
        return list(self.stencil_object.parameter_info.keys())


class FV3StencilObject(StencilWrapper):
    """GT4Py stencil object used for fv3core."""

    def __init__(self, func: Callable[..., None], **kwargs):
        super().__init__(func, delay_compile=True)

        self.times_called: int = 0
        """Number of times this stencil has been called."""

        self.timers = types.SimpleNamespace(call_run=0.0, run=0.0)
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

    def __call__(
        self,
        *args,
        origin: Optional[Tuple[int, ...]] = None,
        domain: Optional[Index3D] = None,
        validate_args: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Call the stencil, compiling the stencil if necessary.

        The stencil needs to be recompiled if any of the following changes
        1. the origin and/or domain
        2. any external value
        3. the function signature or code

        Args:
            domain: Stencil compute domain (required)
            origin: Data index mapped to (0, 0, 0) in the compute domain (required)
        """
        assert origin is not None, "no origin provided at call time"
        assert domain is not None, "no domain provided at call time"

        # Can optimize this by marking stencils that need these
        axis_offsets = fv3core.utils.axis_offsets(spec.grid, origin, domain)

        self.rebuild = global_config.get_rebuild()
        regenerate_stencil = not self.built or self.rebuild

        # Check if we really do need to regenerate
        if not regenerate_stencil:
            axis_offsets_changed = self._check_axis_offsets(axis_offsets)
            regenerate_stencil = regenerate_stencil or axis_offsets_changed

        if self._passed_externals and not regenerate_stencil:
            passed_externals_changed = self._check_passed_externals()
            regenerate_stencil = regenerate_stencil or passed_externals_changed

        if regenerate_stencil:
            self.backend = global_config.get_backend()
            new_build_info: Dict[str, Any] = {}
            stencil_kwargs = {
                "rebuild": self.rebuild,
                "backend": self.backend,
                "externals": {
                    "namelist": spec.namelist,
                    "grid": spec.grid,
                    **axis_offsets,
                    **self._passed_externals,
                },
                "format_source": self.format_source,
                **self.backend_kwargs,
            }
            self._set_device_sync(stencil_kwargs)

            # gtscript.stencil always returns a new class instance even if it
            # used the cached module.
            self.stencil_object = gtscript.stencil(
                definition=self.func, build_info=new_build_info, **stencil_kwargs
            )
            stencil = self.stencil_object
            if stencil not in self._data_cache and "def_ir" in new_build_info:
                def_ir = new_build_info["def_ir"]
                axis_offsets = {
                    k: v for k, v in def_ir.externals.items() if k in axis_offsets
                }
                self._data_cache[stencil] = dict(
                    axis_offsets=axis_offsets, passed_externals=self._passed_externals
                )

        # Call it
        kwargs["exec_info"] = kwargs.get("exec_info", {})
        name = f"{self.func.__module__}.{self.func.__name__}"

        if not self.field_origins or origin != self.origin:
            self.field_origins = self._compute_field_origins(
                origin,
                *args,
                **kwargs,
            )
            self.origin = origin

        if validate_args is None:
            validate_args = global_config.get_validate_args()

        if validate_args:
            _maybe_save_report(
                f"{name}-before",
                self.times_called,
                self.func.__dict__["_gtscript_"]["api_signature"],
                args,
                kwargs,
            )

            self.stencil_object(
                *args,
                **kwargs,
                origin=self.field_origins,
                domain=domain,
                validate_args=True,
            )

            # Update timers
            exec_info = kwargs["exec_info"]
            self.timers.run += exec_info["run_end_time"] - exec_info["run_start_time"]
            self.timers.call_run += (
                exec_info["call_run_end_time"] - exec_info["call_run_start_time"]
            )

            _maybe_save_report(
                f"{name}-after",
                self.times_called,
                self.func.__dict__["_gtscript_"]["api_signature"],
                args,
                kwargs,
            )
            self.times_called += 1
        else:
            kwargs = self._process_kwargs(
                domain,
                *args,
                **kwargs,
            )
            self.stencil_object.run(**kwargs)


class StencilObjectCache:
    """Stencil object cache to enable run time access to compiled stencils."""

    _instance = None

    @classmethod
    def instance(cls):
        """Instance method for Singleton pattern."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.stencil_cache = {}
        return cls._instance

    def __init__(self):
        self.stencil_cache = {}

    def add(self, stencil_object: StencilWrapper):
        stencil_name = str(stencil_object.func)
        self.stencil_cache[stencil_name] = stencil_object

    def clear(self):
        for stencil_object in self.stencil_cache.values():
            stencil_object.clear()


def get_stencil_cache() -> StencilObjectCache:
    return StencilObjectCache.instance()


def gtstencil(**stencil_kwargs) -> Callable[[Any], FV3StencilObject]:
    _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs)

    def decorator(func):
        stencil_object = FV3StencilObject(func, **stencil_kwargs)
        get_stencil_cache().add(stencil_object)
        return stencil_object

    return decorator


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
