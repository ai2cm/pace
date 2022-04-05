import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dace
import gt4py.storage
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map

from pace.dsl.dace.dace_config import dace_config
from pace.dsl.dace.sdfg_opt_passes import strip_unused_global_in_compute_x_flux
from pace.dsl.dace.utils import DaCeProgress


def dace_inhibitor(func: Callable):
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return func


def upload_to_device(host_data: List[Any]):
    """Make sure any data that are still a gt4py.storage gets uploaded to device"""
    for data in host_data:
        if isinstance(data, gt4py.storage.Storage):
            data.host_to_device()


def download_results_from_dace(dace_result: Optional[List[Any]], args: List[Any]):
    """Move all data from DaCe memory space to GT4Py"""
    if dace_result is not None:
        for arg in args:
            if isinstance(arg, gt4py.storage.Storage) and hasattr(
                arg, "_set_device_modified"
            ):
                arg._set_device_modified()
        if dace_config.is_gpu_backend():
            gt4py_results = [
                gt4py.storage.from_array(
                    r,
                    default_origin=(0, 0, 0),
                    backend=dace_config.get_backend(),
                    managed_memory=True,
                )
                for r in dace_result
            ]
        else:
            gt4py_results = [
                gt4py.storage.from_array(
                    r, default_origin=(0, 0, 0), backend=dace_config.get_backend()
                )
                for r in dace_result
            ]
    return gt4py_results


def to_gpu(sdfg: dace.SDFG):
    """Flag memory in SDFG to GPU.
    Force deactivate OpenMP sections for sanity."""

    # Gather all maps
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    # Set storage of arrays to GPU, scalarizable arrays will be set on registers
    for sd, _aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dace.StorageType.Register
        else:
            arr.storage = dace.StorageType.GPU_Global

    # All maps will be scedule on GPU
    for mapentry, state in topmaps:
        mapentry.schedule = dace.ScheduleType.GPU_Device

    # Deactivate OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def call_sdfg(daceprog: DaceProgram, sdfg: dace.SDFG, args, kwargs, sdfg_final=False):
    if not sdfg_final:
        if dace_config.is_gpu_backend():
            to_gpu(sdfg)
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.GPU)
        else:
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.CPU)

    # Make sure all data have been uploaded to the device
    upload_to_device(list(args) + list(kwargs.values()))

    if not sdfg_final:
        sdfg_kwargs = daceprog._create_sdfg_args(sdfg, args, kwargs)
        for k in daceprog.constant_args:
            if k in sdfg_kwargs:
                del sdfg_kwargs[k]
        sdfg_kwargs = {k: v for k, v in sdfg_kwargs.items() if v is not None}
        for k, tup in daceprog.resolver.closure_arrays.items():
            if k in sdfg_kwargs and tup[1].transient:
                del sdfg_kwargs[k]

        # Promote scalar
        from dace.sdfg.analysis import scalar_to_symbol as scal2sym

        with DaCeProgress("Scalar promotion"):
            for sd in sdfg.all_sdfgs_recursive():
                scal2sym.promote_scalars_to_symbols(sd)

        with DaCeProgress("Simplify (1 of 2)"):
            sdfg.simplify(validate=False)

        with DaCeProgress("Expand library nodes (including stencils)"):
            sdfg.expand_library_nodes()

        with DaCeProgress("Simplify (final)"):
            sdfg.simplify(validate=False)

        with DaCeProgress(
            "Refine optimisations:\n  compute_x_flux transient as global removal"
        ):
            strip_unused_global_in_compute_x_flux(sdfg)

        # Call
        res = sdfg(**sdfg_kwargs)
    else:
        res = daceprog(*args, **kwargs)

    return download_results_from_dace(res, list(args) + list(kwargs.values()))


class LazyComputepathFunction(SDFGConvertible):
    """JIT wrapper around a function for DaCe orchestration.

    If use_dace() is False, the wrapper will just return the original callable.

    Attributes:
        func: function to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    def __init__(self, func, load_sdfg):
        self.func = func
        self._load_sdfg = load_sdfg
        self.daceprog = dace.program(self.func)
        self._sdfg_loaded = False
        self._sdfg = None

    def __call__(self, *args, **kwargs):
        if self.use_dace:
            sdfg = self.__sdfg__(*args, **kwargs)
            return call_sdfg(
                self.daceprog,
                sdfg,
                args,
                kwargs,
                sdfg_final=(self._load_sdfg is not None),
            )
        else:
            return self.func(*args, **kwargs)

    @property
    def global_vars(self):
        return self.daceprog.global_vars

    @global_vars.setter
    def global_vars(self, value):
        self.daceprog.global_vars = value

    def __sdfg__(self, *args, **kwargs):
        if self._load_sdfg is None:
            return self.daceprog.to_sdfg(
                *args,
                **self.daceprog.__sdfg_closure__(),
                **kwargs,
                save=False,
                simplify=False,
            )
        else:
            if not self._sdfg_loaded:
                if os.path.isfile(self._load_sdfg):
                    self.daceprog.load_sdfg(self._load_sdfg, *args, **kwargs)
                    self._sdfg_loaded = True
                else:
                    self.daceprog.load_precompiled_sdfg(
                        self._load_sdfg, *args, **kwargs
                    )
                    self._sdfg_loaded = True
            return next(iter(self.daceprog._cache.cache.values())).sdfg

    def __sdfg_closure__(self, *args, **kwargs):
        return self.daceprog.__sdfg_closure__(*args, **kwargs)

    def __sdfg_signature__(self):
        return self.daceprog.argnames, self.daceprog.constant_args

    def closure_resolver(self, constant_args, given_args, parent_closure=None):
        return self.daceprog.closure_resolver(constant_args, given_args, parent_closure)

    @property
    def use_dace(self):
        return dace_config.get_orchestrate()


class LazyComputepathMethod:
    """JIT wrapper around a class method for DaCe orchestration.

    If use_dace() is False, the wrapper will just return the original callable.

    Attributes:
        method: class method to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    # In order to not regenerate SDFG for the same obj.method callable
    # we cache the SDFGEnabledCallable we have already init
    bound_callables: Dict[Tuple[int, int], "SDFGEnabledCallable"] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(self, lazy_method: "LazyComputepathMethod", obj_to_bind):
            methodwrapper = dace.method(lazy_method.func)
            self.obj_to_bind = obj_to_bind
            self.lazy_method = lazy_method
            self.daceprog = methodwrapper.__get__(obj_to_bind)

        @property
        def global_vars(self):
            return self.daceprog.global_vars

        @global_vars.setter
        def global_vars(self, value):
            self.daceprog.global_vars = value

        def __call__(self, *args, **kwargs):
            if self.lazy_method.use_dace:
                sdfg = self.__sdfg__(*args, **kwargs)
                return call_sdfg(
                    self.daceprog,
                    sdfg,
                    args,
                    kwargs,
                    sdfg_final=(self.lazy_method._load_sdfg is not None),
                )
            else:
                return self.lazy_method.func(self.obj_to_bind, *args, **kwargs)

        def __sdfg__(self, *args, **kwargs):
            if self.lazy_method._load_sdfg is None:
                return self.daceprog.to_sdfg(
                    *args,
                    **self.daceprog.__sdfg_closure__(),
                    **kwargs,
                    save=False,
                    simplify=False,
                )
            else:
                if os.path.isfile(self.lazy_method._load_sdfg):
                    self.daceprog.load_sdfg(
                        self.lazy_method._load_sdfg, *args, **kwargs
                    )
                else:
                    self.daceprog.load_precompiled_sdfg(
                        self.lazy_method._load_sdfg, *args, **kwargs
                    )
                return self.daceprog.__sdfg__(*args, **kwargs)

        def __sdfg_closure__(self, reevaluate=None):
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, given_args, parent_closure=None):
            return self.daceprog.closure_resolver(
                constant_args, given_args, parent_closure
            )

    def __init__(self, func, load_sdfg):
        self.func = func
        self._load_sdfg = load_sdfg

    def __get__(self, obj, objype=None) -> SDFGEnabledCallable:
        """Return SDFGEnabledCallable wrapping original obj.method from cache.
        Update cache first if need be"""
        if (id(obj), id(self.func)) not in LazyComputepathMethod.bound_callables:

            LazyComputepathMethod.bound_callables[
                (id(obj), id(self.func))
            ] = LazyComputepathMethod.SDFGEnabledCallable(self, obj)

        return LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]

    @property
    def use_dace(self):
        return dace_config.get_orchestrate()


def computepath_method(
    method: Callable[..., Any], load_sdfg: Optional[str] = None
) -> Union[Callable[..., Any], LazyComputepathFunction]:
    """
    Decorator wrapping a class method in a JIT DaCe orchestrator.

    Args:
        method: class method to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
            that will be compiled but not regenerated."""

    def _decorator(method, load_sdfg):
        return LazyComputepathMethod(method, load_sdfg)

    return _decorator(method, load_sdfg)


def computepath_function(
    function: Callable[..., Any],
    load_sdfg: Optional[str] = None,
) -> Union[Callable[..., Any], LazyComputepathFunction]:
    """
    Decorator wrapping a function in a JIT DaCe orchestrator.

    Args:
        method: class method to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
            that will be compiled but not regenerated."""

    def _decorator(function, load_sdfg):
        return LazyComputepathFunction(function, load_sdfg)

    return _decorator(function, load_sdfg)
