import os
from typing import Dict, Tuple, Callable, Union, Any
import inspect

import dace
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map

from pace.dsl.dace.sdfg_opt_passes import refine_permute_arrays
from pace.dsl.dace.dace_config import dace_config

import gt4py.storage


def dace_inhibitor(fn: Callable):
    """Triggers callback generation wrapping `fn` while doing DaCe parsing."""
    return fn


def to_gpu(sdfg: dace.SDFG):
    """Move all arrays to GPU, distributing between global memory and register
    based on the shape"""
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    for sd, _aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dace.StorageType.Register
        else:
            arr.storage = dace.StorageType.GPU_Global

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
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, gt4py.storage.Storage):
            arg.host_to_device()

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

        for sd in sdfg.all_sdfgs_recursive():
            scal2sym.promote_scalars_to_symbols(sd)

        # Simplify the SDFG (automatic optimization)
        sdfg.simplify(validate=False)

        # Here we insert optimization passes that don't exists in Simplify yet
        refine_permute_arrays(sdfg)

        # Call
        res = sdfg(**sdfg_kwargs)
    else:
        res = daceprog(*args, **kwargs)
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, gt4py.storage.Storage) and hasattr(
            arg, "_set_device_modified"
        ):
            arg._set_device_modified()
    if res is not None:
        if dace_config.is_gpu_backend():
            res = [
                gt4py.storage.from_array(
                    r,
                    default_origin=(0, 0, 0),
                    backend=dace_config.get_backend(),
                    managed_memory=True,
                )
                for r in res
            ]
        else:
            res = [
                gt4py.storage.from_array(
                    r, default_origin=(0, 0, 0), backend=dace_config.get_backend()
                )
                for r in res
            ]
    return res


class LazyComputepathFunction:
    def __init__(self, func, use_dace, skip_dacemode, load_sdfg):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
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
        return self._use_dace or (
            dace_config.get_orchestrate() and not self._skip_dacemode
        )


class LazyComputepathMethod:

    bound_callables: Dict[Tuple[int, int], Callable] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(self, lazy_method, obj_to_bind):
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

    def __init__(self, func, use_dace, skip_dacemode, load_sdfg, arg_spec):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
        self._load_sdfg = load_sdfg
        self.arg_spec = arg_spec

    def __get__(self, obj, objype=None):

        if (id(obj), id(self.func)) not in LazyComputepathMethod.bound_callables:

            LazyComputepathMethod.bound_callables[
                (id(obj), id(self.func))
            ] = LazyComputepathMethod.SDFGEnabledCallable(self, obj)

        return LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]

    @property
    def use_dace(self):
        return self._use_dace or (
            dace_config.get_orchestrate() and not self._skip_dacemode
        )


def computepath_method(*args, **kwargs):
    skip_dacemode = kwargs.get("skip_dacemode", False)
    load_sdfg = kwargs.get("load_sdfg", None)
    use_dace = kwargs.get("use_dace", False)
    arg_spec = inspect.getfullargspec(args[0]) if len(args) == 1 else None

    def _decorator(method):
        return LazyComputepathMethod(
            method, use_dace, skip_dacemode, load_sdfg, arg_spec
        )

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator


def computepath_function(
    *args, **kwargs
) -> Union[Callable[..., Any], LazyComputepathFunction]:
    skip_dacemode = kwargs.get("skip_dacemode", False)
    load_sdfg = kwargs.get("load_sdfg", None)
    use_dace = kwargs.get("use_dace", False)

    def _decorator(function):
        return LazyComputepathFunction(function, use_dace, skip_dacemode, load_sdfg)

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator
