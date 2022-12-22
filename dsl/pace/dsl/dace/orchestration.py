import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dace
import gt4py.storage
from dace import compiletime as DaceCompiletime
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import StorageType as DaceStorageType
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map

from pace.dsl.dace.build import (
    determine_compiling_ranks,
    get_sdfg_path,
    unblock_waiting_tiles,
    write_build_info,
)
from pace.dsl.dace.dace_config import (
    DEACTIVATE_DISTRIBUTED_DACE_COMPILE,
    DaceConfig,
    DaCeOrchestration,
)
from pace.dsl.dace.sdfg_debug_passes import (
    negative_delp_checker,
    negative_qtracers_checker,
    trace_all_outputs_at_index,
)
from pace.dsl.dace.sdfg_opt_passes import splittable_region_expansion
from pace.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
)
from pace.util.mpi import MPI


try:
    import cupy as cp
except ImportError:
    cp = None


def dace_inhibitor(func: Callable):
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return func


def _upload_to_device(host_data: List[Any]):
    """Make sure any ndarrays gets uploaded to the device

    This will raise an assertion if cupy is not installed.
    """
    assert cp is not None
    for i, data in enumerate(host_data):
        if isinstance(data, cp.ndarray):
            host_data[i] = cp.asarray(data)


def _download_results_from_dace(
    config: DaceConfig, dace_result: Optional[List[Any]], args: List[Any]
):
    """Move all data from DaCe memory space to GT4Py"""
    gt4py_results = None
    if dace_result is not None:
        if config.is_gpu_backend():
            gt4py_results = [
                gt4py.storage.from_array(
                    r,
                    default_origin=(0, 0, 0),
                    backend=config.get_backend(),
                    managed_memory=True,
                )
                for r in dace_result
            ]
        else:
            gt4py_results = [
                gt4py.storage.from_array(
                    r, default_origin=(0, 0, 0), backend=config.get_backend()
                )
                for r in dace_result
            ]
    return gt4py_results


def _to_gpu(sdfg: dace.SDFG):
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


def _run_sdfg(daceprog: DaceProgram, config: DaceConfig, args, kwargs):
    """Execute a compiled SDFG - do not check for compilation"""
    if config.is_gpu_backend():
        _upload_to_device(list(args) + list(kwargs.values()))
    res = daceprog(*args, **kwargs)
    return _download_results_from_dace(config, res, list(args) + list(kwargs.values()))


def _build_sdfg(
    daceprog: DaceProgram, sdfg: dace.SDFG, config: DaceConfig, args, kwargs
):
    """Build the .so out of the SDFG on the top tile ranks only"""
    if DEACTIVATE_DISTRIBUTED_DACE_COMPILE:
        is_compiling = True
    else:
        is_compiling = determine_compiling_ranks(config)
    if is_compiling:
        # Make the transients array persistents
        if config.is_gpu_backend():
            _to_gpu(sdfg)
            make_transients_persistent(sdfg=sdfg, device=DaceDeviceType.GPU)

            # Upload args to device
            _upload_to_device(list(args) + list(kwargs.values()))
        else:
            for sd, _aname, arr in sdfg.arrays_recursive():
                if arr.shape == (1,):
                    arr.storage = DaceStorageType.Register
            make_transients_persistent(sdfg=sdfg, device=DaceDeviceType.CPU)

        # Build non-constants & non-transients from the sdfg_kwargs
        sdfg_kwargs = daceprog._create_sdfg_args(sdfg, args, kwargs)
        for k in daceprog.constant_args:
            if k in sdfg_kwargs:
                del sdfg_kwargs[k]
        sdfg_kwargs = {k: v for k, v in sdfg_kwargs.items() if v is not None}
        for k, tup in daceprog.resolver.closure_arrays.items():
            if k in sdfg_kwargs and tup[1].transient:
                del sdfg_kwargs[k]

        with DaCeProgress(config, "Simplify (1/2)"):
            sdfg.simplify(validate=False, verbose=True)

        # Perform pre-expansion fine tuning
        with DaCeProgress(config, "Split regions"):
            splittable_region_expansion(sdfg, verbose=True)

        # Expand the stencil computation Library Nodes with the right expansion
        with DaCeProgress(config, "Expand"):
            sdfg.expand_library_nodes()

        with DaCeProgress(config, "Simplify (2/2)"):
            sdfg.simplify(validate=False, verbose=True)

        # Move all memory that can be into a pool to lower memory pressure.
        # Change Persistent memory (sub-SDFG) into Scope and flag it.
        with DaCeProgress(config, "Turn Persistents into pooled Scope"):
            memory_pooled = 0.0
            for _sd, _aname, arr in sdfg.arrays_recursive():
                if arr.lifetime == dace.AllocationLifetime.Persistent:
                    arr.pool = True
                    memory_pooled += arr.total_size * arr.dtype.bytes
                    arr.lifetime = dace.AllocationLifetime.Scope
            memory_pooled = float(memory_pooled) / (1024 * 1024)
            DaCeProgress.log(
                DaCeProgress.default_prefix(config),
                f"Pooled {memory_pooled} mb",
            )

        # Set of debug tools inserted in the SDFG when dace.conf "syncdebug"
        # is turned on.
        if config.get_sync_debug():
            with DaCeProgress(config, "Tooling the SDFG for debug"):
                # sdfg_nan_checker(sdfg) # TODO (florian): segfault - bad range?
                trace_all_outputs_at_index(sdfg, 0, 0, 60)
                negative_delp_checker(sdfg)
                negative_qtracers_checker(sdfg)

        # Compile
        with DaCeProgress(config, "Codegen & compile"):
            sdfg.compile()
        write_build_info(sdfg, config.layout, config.tile_resolution, config._backend)

        # Printing analysis of the compiled SDFG
        with DaCeProgress(config, "Build finished. Running memory static analysis"):
            DaCeProgress.log(
                DaCeProgress.default_prefix(config),
                report_memory_static_analysis(
                    sdfg, memory_static_analysis(sdfg), False
                ),
            )

    # Compilation done, either exit or scatter/gather and run
    # DEV NOTE: we explicitly use MPI.COMM_WORLD here because it is
    # a true multi-machine sync, outside of our own communicator class.
    # Also this code is protected in the case of running on one machine by the fact
    # that 0 is _always_ a compiling rank & unblock_waiting_tiles is protected
    # against scattering when no other ranks are present.
    if config.get_orchestrate() == DaCeOrchestration.Build:
        MPI.COMM_WORLD.Barrier()  # Protect against early exist which kill SLURM jobs
        DaCeProgress.log(
            DaCeProgress.default_prefix(config),
            "Build only, exiting.",
        )
        exit(0)
    elif config.get_orchestrate() == DaCeOrchestration.BuildAndRun:
        MPI.COMM_WORLD.Barrier()
        if is_compiling:
            if not DEACTIVATE_DISTRIBUTED_DACE_COMPILE:
                unblock_waiting_tiles(MPI.COMM_WORLD, sdfg.build_folder)
                DaCeProgress.log(
                    DaCeProgress.default_prefix(config), "Build folder exchanged."
                )
            with DaCeProgress(config, "Run"):
                res = sdfg(**sdfg_kwargs)
                res = _download_results_from_dace(
                    config, res, list(args) + list(kwargs.values())
                )
        else:
            source_rank = config.target_rank
            # wait for compilation to be done
            DaCeProgress.log(
                DaCeProgress.default_prefix(config),
                "Rank is not compiling. Waiting for build dir...",
            )
            sdfg_path = MPI.COMM_WORLD.recv(source=source_rank)
            DaCeProgress.log(DaCeProgress.default_prefix(config), "Build dir received.")
            daceprog.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
            with DaCeProgress(config, "Run"):
                res = _run_sdfg(daceprog, config, args, kwargs)

        return res


def _call_sdfg(
    daceprog: DaceProgram, sdfg: dace.SDFG, config: DaceConfig, args, kwargs
):
    """Dispatch the SDFG execution and/or build"""
    if (
        config.get_orchestrate() == DaCeOrchestration.Build
        or config.get_orchestrate() == DaCeOrchestration.BuildAndRun
    ):
        return _build_sdfg(daceprog, sdfg, config, args, kwargs)
    elif config.get_orchestrate() == DaCeOrchestration.Run:
        with DaCeProgress(config, "Run"):
            res = _run_sdfg(daceprog, config, args, kwargs)
        return res
    else:
        raise NotImplementedError(
            f"Mode {config.get_orchestrate()} unimplemented at call time"
        )


def _parse_sdfg(
    daceprog: DaceProgram,
    config: DaceConfig,
    *args,
    **kwargs,
) -> Optional[dace.SDFG]:
    """Return an SDFG depending on cache existence.
    Either parses, load a .sdfg or load .so (as a compiled sdfg)

    Attributes:
        daceprog: the DaceProgram carrying reference to the original method/function
        config: the DaceConfig configuration for this execution
    """
    sdfg_path = get_sdfg_path(daceprog.name, config)
    if sdfg_path is None:
        if DEACTIVATE_DISTRIBUTED_DACE_COMPILE:
            is_compiling = True
        else:
            is_compiling = determine_compiling_ranks(config)
        if not is_compiling:
            # We can not parse the SDFG since we will load the proper
            # compiled SDFG from the compiling rank
            return None
        with DaCeProgress(config, f"Parse code of {daceprog.name} to SDFG"):
            sdfg = daceprog.to_sdfg(
                *args,
                **daceprog.__sdfg_closure__(),
                **kwargs,
                save=False,
                simplify=False,
            )
        return sdfg
    else:
        if os.path.isfile(sdfg_path):
            with DaCeProgress(config, "Load .sdfg"):
                sdfg, _ = daceprog.load_sdfg(sdfg_path, *args, **kwargs)
            return sdfg
        else:
            with DaCeProgress(config, "Load precompiled .sdfg (.so)"):
                csdfg, _ = daceprog.load_precompiled_sdfg(sdfg_path, *args, **kwargs)
            return csdfg


class _LazyComputepathFunction(SDFGConvertible):
    """JIT wrapper around a function for DaCe orchestration.

    Attributes:
        func: function to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    def __init__(self, func: Callable, config: DaceConfig):
        self.func = func
        self.config = config
        self.daceprog: DaceProgram = dace.program(self.func)
        self._sdfg = None

    def __call__(self, *args, **kwargs):
        assert self.config.is_dace_orchestrated()
        sdfg = _parse_sdfg(
            self.daceprog,
            self.config,
            *args,
            **kwargs,
        )
        return _call_sdfg(
            self.daceprog,
            sdfg,
            self.config,
            args,
            kwargs,
        )

    @property
    def global_vars(self):
        return self.daceprog.global_vars

    @global_vars.setter
    def global_vars(self, value):
        self.daceprog.global_vars = value

    def __sdfg__(self, *args, **kwargs):
        return _parse_sdfg(self.daceprog, self.config, *args, **kwargs)

    def __sdfg_closure__(self, *args, **kwargs):
        return self.daceprog.__sdfg_closure__(*args, **kwargs)

    def __sdfg_signature__(self):
        return self.daceprog.argnames, self.daceprog.constant_args

    def closure_resolver(self, constant_args, given_args, parent_closure=None):
        return self.daceprog.closure_resolver(constant_args, given_args, parent_closure)


class _LazyComputepathMethod:
    """JIT wrapper around a class method for DaCe orchestration.

    Attributes:
        method: class method to either orchestrate or directly execute
        load_sdfg: folder path to a pre-compiled SDFG or file path to a .sdfg graph
                   that will be compiled but not regenerated.
    """

    # In order to not regenerate SDFG for the same obj.method callable
    # we cache the SDFGEnabledCallable we have already init
    bound_callables: Dict[Tuple[int, int], "SDFGEnabledCallable"] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(self, lazy_method: "_LazyComputepathMethod", obj_to_bind):
            methodwrapper = dace.method(lazy_method.func)
            self.obj_to_bind = obj_to_bind
            self.lazy_method = lazy_method
            self.daceprog: DaceProgram = methodwrapper.__get__(obj_to_bind)

        @property
        def global_vars(self):
            return self.daceprog.global_vars

        @global_vars.setter
        def global_vars(self, value):
            self.daceprog.global_vars = value

        def __call__(self, *args, **kwargs):
            assert self.lazy_method.config.is_dace_orchestrated()
            sdfg = _parse_sdfg(
                self.daceprog,
                self.lazy_method.config,
                *args,
                **kwargs,
            )
            return _call_sdfg(
                self.daceprog,
                sdfg,
                self.lazy_method.config,
                args,
                kwargs,
            )

        def __sdfg__(self, *args, **kwargs):
            return _parse_sdfg(self.daceprog, self.lazy_method.config, *args, **kwargs)

        def __sdfg_closure__(self, reevaluate=None):
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, given_args, parent_closure=None):
            return self.daceprog.closure_resolver(
                constant_args, given_args, parent_closure
            )

    def __init__(self, func: Callable, config: DaceConfig):
        self.func = func
        self.config = config

    def __get__(self, obj, objtype=None) -> SDFGEnabledCallable:
        """Return SDFGEnabledCallable wrapping original obj.method from cache.
        Update cache first if need be"""
        if (id(obj), id(self.func)) not in _LazyComputepathMethod.bound_callables:

            _LazyComputepathMethod.bound_callables[
                (id(obj), id(self.func))
            ] = _LazyComputepathMethod.SDFGEnabledCallable(self, obj)

        return _LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]


def orchestrate(
    *,
    obj: object,
    config: DaceConfig,
    method_to_orchestrate: str = "__call__",
    dace_compiletime_args: Optional[Sequence[str]] = None,
):
    """
    Orchestrate a method of an object with DaCe.
    The method object is patched in place, replacing the orignal Callable with
    a wrapper that will trigger orchestration at call time.
    If the model configuration doesn't demand orchestration, this won't do anything.

    Args:
        obj: object which methods is to be orchestrated
        config: DaceConfig carrying model configuration
        method_to_orchestrate: string representing the name of the method
        dace_compiletime_args: list of names of arguments to be flagged has
                               dace.compiletime for orchestration to behave
    """
    if dace_compiletime_args is None:
        dace_compiletime_args = []

    if config.is_dace_orchestrated():
        if hasattr(obj, method_to_orchestrate):
            func = type.__getattribute__(type(obj), method_to_orchestrate)

            # Flag argument as dace.constant
            for argument in dace_compiletime_args:
                func.__annotations__[argument] = DaceCompiletime

            # Build DaCe orchestrated wrapper
            # This is a JIT object, e.g. DaCe compilation will happen on call
            wrapped = _LazyComputepathMethod(func, config).__get__(obj)

            if method_to_orchestrate == "__call__":
                # Grab the function from the type of the child class
                # Dev note: we need to use type for dunder call because:
                #   a = A()
                #   a()
                # resolved to: type(a).__call__(a)
                # therefore patching the instance call (e.g a.__call__) is not enough.
                # We could patch the type(self), ergo the class itself
                # but that would patch _every_ instance of A.
                # What we can do is patch the instance.__class__ with a local made class
                # in order to keep each instance with it's own patch.
                #
                # Re: type:ignore
                # Mypy is unhappy about dynamic class name and the devs (per github
                # issues discussion) is to make a plugin. Too much work -> ignore mypy

                class _(type(obj)):  # type: ignore
                    __qualname__ = f"{type(obj).__qualname__}_patched"
                    __name__ = f"{type(obj).__name__}_patched"

                    def __call__(self, *arg, **kwarg):
                        return wrapped(*arg, **kwarg)

                    def __sdfg__(self, *args, **kwargs):
                        return wrapped.__sdfg__(*args, **kwargs)

                    def __sdfg_closure__(self, reevaluate=None):
                        return wrapped.__sdfg_closure__(reevaluate)

                    def __sdfg_signature__(self):
                        return wrapped.__sdfg_signature__()

                    def closure_resolver(
                        self, constant_args, given_args, parent_closure=None
                    ):
                        return wrapped.closure_resolver(
                            constant_args, given_args, parent_closure
                        )

                # We keep the original class type name to not perturb
                # the workflows that uses it to build relevant info (path, hash...)
                previous_cls_name = type(obj).__name__
                obj.__class__ = _
                type(obj).__name__ = previous_cls_name
            else:
                # For regular attribute - we can just patch as usual
                setattr(obj, method_to_orchestrate, wrapped)

        else:
            raise RuntimeError(
                f"Could not orchestrate, "
                f"{type(obj).__name__}.{method_to_orchestrate} "
                "does not exists"
            )


def orchestrate_function(
    config: DaceConfig = None,
    dace_compiletime_args: Optional[Sequence[str]] = None,
) -> Union[Callable[..., Any], _LazyComputepathFunction]:
    """
    Decorator orchestrating a method of an object with DaCe.
    If the model configuration doesn't demand orchestration, this won't do anything.

    Args:
        config: DaceConfig carrying model configuration
        dace_compiletime_args: list of names of arguments to be flagged has
                               dace.compiletime for orchestration to behave
    """

    if dace_compiletime_args is None:
        dace_compiletime_args = []

    def _decorator(func: Callable[..., Any]):
        def _wrapper(*args, **kwargs):
            for argument in dace_compiletime_args:
                func.__annotations__[argument] = DaceCompiletime
            return _LazyComputepathFunction(func, config)

        if config.is_dace_orchestrated():
            return _wrapper(func)
        else:
            return func

    return _decorator
