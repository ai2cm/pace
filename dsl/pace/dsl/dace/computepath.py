import inspect
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import dace
import gt4py.storage
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map
from dace.transformation.transformation import simplification_transformations

from pace.util import global_config
from pace.dsl.dace.build import (
    determine_compiling_ranks,
    load_sdfg_once,
    read_target_rank,
    unblock_waiting_tiles,
    write_decomposition,
)
from pace.dsl.dace.sdfg_opt_passes import (
    splittable_region_expansion,
    strip_unused_global_in_compute_x_flux,
)
from pace.dsl.dace.utils import DaCeProgress
from pace.util.mpi import MPI


_CONSTANT_PROPAGATION = False


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
        elif isinstance(arr.dtype, dace.opaque):
            continue
        else:
            arr.storage = dace.StorageType.GPU_Global

    for mapentry, _state in topmaps:
        mapentry.schedule = dace.ScheduleType.GPU_Device

    # Deactivate OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def upload_to_device(host_data: List[Any]):
    """Make sure any data that are still a gt4py.storage gets uploaded to device"""
    for data in host_data:
        if isinstance(data, gt4py.storage.Storage):
            data.host_to_device()


def download_results_from_dace(res, args):
    for arg in args:
        if isinstance(arg, gt4py.storage.Storage) and hasattr(
            arg, "_set_device_modified"
        ):
            arg._set_device_modified()
    if res is not None:
        if global_config.is_gpu_backend():
            res = [
                gt4py.storage.from_array(
                    r,
                    default_origin=(0, 0, 0),
                    backend=global_config.get_backend(),
                    managed_memory=True,
                )
                for r in res
            ]
        else:
            res = [
                gt4py.storage.from_array(
                    r, default_origin=(0, 0, 0), backend=global_config.get_backend()
                )
                for r in res
            ]
    return res


def run_sdfg(daceprog: DaceProgram, sdfg: dace.SDFG, args, kwargs):
    """Production mode: .so should exist and be"""
    upload_to_device(list(args) + list(kwargs.values()))
    res = daceprog(*args, **kwargs)
    return download_results_from_dace(res, list(args) + list(kwargs.values()))


def build_sdfg(daceprog: DaceProgram, sdfg: dace.SDFG, args, kwargs):
    is_compiling, comm = determine_compiling_ranks()
    if is_compiling:
        if comm.Get_rank() == 0 and comm.Get_size() > 1:
            write_decomposition()
        # Make the transients array persistents
        if global_config.is_gpu_backend():
            to_gpu(sdfg)
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.GPU)
        else:
            for sd, _aname, arr in sdfg.arrays_recursive():
                if arr.shape == (1,):
                    arr.storage = dace.StorageType.Register
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.CPU)

        # Upload args to device
        upload_to_device(list(args) + list(kwargs.values()))

        # Build non-constants & non-transients from the sdfg_kwargs
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

        if _CONSTANT_PROPAGATION:
            # Constant propagation - part 1
            with DaCeProgress("Constant propagation"):
                from pace.dsl.dace.sdfg_opt_passes import simple_cprop

                for sd in sdfg.all_sdfgs_recursive():
                    simple_cprop(sd)

        if _CONSTANT_PROPAGATION:
            # Constant propagation - part 2
            from dace.transformation.interstate.state_elimination import (
                DeadStateElimination,
                FalseConditionElimination,
            )

            with DaCeProgress("Simplify + DeadState / FalsCond elimination"):

                sdfg.apply_transformations_repeated(
                    [DeadStateElimination, FalseConditionElimination]
                    + simplification_transformations()
                )
        else:
            with DaCeProgress("Simplify (1 of 2)"):
                sdfg.simplify(validate=False)

        # Perform pre-expansion fine tuning
        splittable_region_expansion(sdfg)

        # Expand the stencil computation Library Nodes with the right expansion
        with DaCeProgress("Expand"):
            sdfg.expand_library_nodes()

        # Simplify again after expansion
        with DaCeProgress("Simplify (final)"):
            sdfg.simplify(validate=False)

        with DaCeProgress(
            "Removed of compute_x_flux transients (to lower VRAM and because of their evilness)"
        ):
            strip_unused_global_in_compute_x_flux(sdfg)

        # Compile
        with DaCeProgress("Codegen & compile"):
            sdfg.compile()

    # Compilation done, either exit or scatter/gather and run
    if global_config.get_dacemode() == global_config.DaCeOrchestration.Build:
        MPI.COMM_WORLD.Barrier()  # Protect against early exist which kill SLURM jobs
        DaCeProgress.log("Compilation finished and saved, exiting.")
        exit(0)
    elif global_config.get_dacemode() == global_config.DaCeOrchestration.BuildAndRun:
        MPI.COMM_WORLD.Barrier()
        if is_compiling:
            unblock_waiting_tiles(comm, sdfg.build_folder)
            with DaCeProgress("Run"):
                res = sdfg(**sdfg_kwargs)
                res = download_results_from_dace(
                    res, list(args) + list(kwargs.values())
                )
        else:
            from gt4py import config as gt_config

            config_path = (
                f"{gt_config.cache_settings['root_path']}/.layout/decomposition.yml"
            )
            source_rank = read_target_rank(comm.Get_rank(), config_path)
            # wait for compilation to be done
            sdfg_path = comm.recv(source=source_rank)
            daceprog.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
            with DaCeProgress("Run"):
                res = run_sdfg(daceprog, sdfg, args, kwargs)

        return res


def call_sdfg(daceprog: DaceProgram, sdfg: dace.SDFG, args, kwargs):
    if (
        global_config.get_dacemode() == global_config.DaCeOrchestration.Build
        or global_config.get_dacemode() == global_config.DaCeOrchestration.BuildAndRun
    ):
        return build_sdfg(daceprog, sdfg, args, kwargs)
    elif global_config.get_dacemode() == global_config.DaCeOrchestration.Run:
        return run_sdfg(daceprog, sdfg, args, kwargs)
    else:
        raise NotImplementedError(
            f"Mode {global_config.get_dacemode()} unimplemented at call time"
        )


class LazyComputepathFunction:
    def __init__(self, func, use_dace, skip_dacemode):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
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
        sdfg_path = load_sdfg_once(self.func)
        if not self._sdfg_loaded and sdfg_path is None:
            return self.daceprog.to_sdfg(
                *args,
                **self.daceprog.__sdfg_closure__(),
                **kwargs,
                save=False,
                simplify=False,
            )
        else:
            if not self._sdfg_loaded:
                if os.path.isfile(sdfg_path):
                    self.daceprog.load_sdfg(sdfg_path, *args, **kwargs)
                    self._sdfg_loaded = True
                else:
                    self.daceprog.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
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
            global_config.is_dace_orchestrated() and not self._skip_dacemode
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
                )
            else:
                return self.lazy_method.func(self.obj_to_bind, *args, **kwargs)

        def __sdfg__(self, *args, **kwargs):
            sdfg_path = load_sdfg_once(self.lazy_method.func)
            if sdfg_path is None:
                return self.daceprog.to_sdfg(
                    *args,
                    **self.daceprog.__sdfg_closure__(),
                    **kwargs,
                    save=False,
                    simplify=False,
                )
            else:
                if os.path.isfile(sdfg_path):
                    self.daceprog.load_sdfg(sdfg_path, *args, **kwargs)
                else:
                    self.daceprog.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
                return self.daceprog.__sdfg__(*args, **kwargs)

        def __sdfg_closure__(self, reevaluate=None):
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, given_args, parent_closure=None):
            return self.daceprog.closure_resolver(
                constant_args, given_args, parent_closure
            )

    def __init__(self, func, use_dace, skip_dacemode, arg_spec):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
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
            global_config.is_dace_orchestrated() and not self._skip_dacemode
        )


def computepath_method(*args, **kwargs):
    skip_dacemode = kwargs.get("skip_dacemode", False)
    use_dace = kwargs.get("use_dace", False)
    arg_spec = inspect.getfullargspec(args[0]) if len(args) == 1 else None

    def _decorator(method):
        return LazyComputepathMethod(method, use_dace, skip_dacemode, arg_spec)

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator


def computepath_function(
    *args, **kwargs
) -> Union[Callable[..., Any], LazyComputepathFunction]:
    skip_dacemode = kwargs.get("skip_dacemode", False)
    use_dace = kwargs.get("use_dace", False)

    def _decorator(function):
        return LazyComputepathFunction(function, use_dace, skip_dacemode)

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator
