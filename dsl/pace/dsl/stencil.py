import copy
import dataclasses
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import dace
import gt4py
import numpy as np
from gt4py import gtscript
from gt4py.storage.storage import Storage
from gtc.passes.oir_pipeline import DefaultPipeline, OirPipeline

import pace.dsl.gt4py_utils as gt4py_utils
import pace.util
from pace.dsl.dace.orchestration import SDFGConvertible
from pace.dsl.stencil_config import CompilationConfig, RunMode, StencilConfig
from pace.dsl.typing import Index3D, cast_to_index3d
from pace.util import testing
from pace.util.decomposition import block_waiting_for_compilation, unblock_waiting_tiles
from pace.util.halo_data_transformer import QuantityHaloSpec
from pace.util.mpi import MPI


def report_difference(args, kwargs, args_copy, kwargs_copy, function_name, gt_id):
    report_head = f"comparing against numpy for func {function_name}, gt_id {gt_id}:"
    report_segments = []
    for i, (arg, numpy_arg) in enumerate(zip(args, args_copy)):
        if isinstance(arg, pace.util.Quantity):
            arg = arg.storage
            numpy_arg = numpy_arg.storage
        if isinstance(arg, np.ndarray):
            report_segments.append(report_diff(arg, numpy_arg, label=f"arg {i}"))
    for name in kwargs:
        if isinstance(kwargs[name], pace.util.Quantity):
            kwarg = kwargs[name].data
            numpy_kwarg = kwargs_copy[name].data
        else:
            kwarg = kwargs[name]
            numpy_kwarg = kwargs_copy[name]
        if isinstance(kwarg, np.ndarray):
            report_segments.append(
                report_diff(kwarg, numpy_kwarg, label=f"kwarg {name}")
            )
    report_body = "".join(report_segments)
    if len(report_body) > 0:
        print("")  # newline
        print(report_head + report_body)


def report_diff(arg: np.ndarray, numpy_arg: np.ndarray, label) -> str:
    metric_err = testing.compare_arr(arg, numpy_arg)
    nans_match = np.logical_and(np.isnan(arg), np.isnan(numpy_arg))
    n_points = np.product(arg.shape)
    failures_14 = n_points - np.sum(
        np.logical_or(
            nans_match,
            metric_err < 1e-14,
        )
    )
    failures_10 = n_points - np.sum(
        np.logical_or(
            nans_match,
            metric_err < 1e-10,
        )
    )
    failures_8 = n_points - np.sum(
        np.logical_or(
            nans_match,
            metric_err < 1e-8,
        )
    )
    greatest_error = np.max(metric_err[~np.isnan(metric_err)])
    if greatest_error == 0.0 and failures_14 == 0:
        report = ""
    else:
        report = f"\n    {label}: "
        report += f"max_err={greatest_error}"
        if failures_14 > 0:
            report += f" 1e-14 failures: {failures_14}"
        if failures_10 > 0:
            report += f" 1e-10 failures: {failures_10}"
        if failures_8 > 0:
            report += f" 1e-8 failures: {failures_8}"
    return report


@dataclasses.dataclass
class TimingCollector:
    """
    Attributes:
        build_info: contains info about the generation process for each stencil.
        exec_info: contains info about the execution of each stencil.
    """

    build_info: Dict[str, dict] = dataclasses.field(default_factory=dict)
    exec_info: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"__aggregate_data": True}
    )

    def build_report(self, key: str = "build_time", **kwargs) -> str:
        return type(self)._show_report(
            self.build_info, self.build_info.keys(), key, **kwargs
        )

    def exec_report(self, key: str = "total_run_time", **kwargs) -> str:
        # NOTE: Uses the build_info keys to distinguish stencils
        return type(self)._show_report(
            self.exec_info, self.build_info.keys(), key, **kwargs
        )

    @staticmethod
    def _show_report(
        infos: Dict[str, Any],
        keys: Iterable[str],
        secondary_key: str,
        *,
        name_width: int = 40,
        bar_width: int = 40,
        delimiter: str = " | ",
        show_bar: bool = True,
        reverse: bool = True,
        digits: int = 3,
    ) -> str:
        assert name_width > 10

        data = [(key, infos[key][secondary_key]) for key in keys]
        sorted_data = tuple(
            sorted(data, key=lambda name_time: name_time[1], reverse=reverse)
        )
        max_val = sorted_data[0 if reverse else -1][1]

        format = f".{digits}e"

        outputs: List[str] = [f"Total: {sum(d[1] for d in data):{format}}"]
        for name, val in sorted_data:
            if len(name) > name_width:
                width = int(name_width / 2) - 3
                disp_name = f"{name[:width]}...{name[-width:]:{format}}"
            else:
                disp_name = name
            line = f"{disp_name.rjust(name_width)}{delimiter}{val:{format}}"
            if show_bar and max_val > 0:
                bar_data = bar = "█" * int(val / max_val * bar_width)
                line += f"{delimiter}{bar_data}"
            outputs.append(line)

        return "\n".join(outputs)


class CompareToNumpyStencil:
    """
    A wrapper over FrozenStencil which executes a numpy version of the stencil as well,
    and compares the results.
    """

    def __init__(
        self,
        func: Callable[..., None],
        origin: Union[Tuple[int, ...], Mapping[str, Tuple[int, ...]]],
        domain: Tuple[int, ...],
        stencil_config: StencilConfig,
        externals: Optional[Mapping[str, Any]] = None,
        skip_passes: Optional[Tuple[str, ...]] = None,
        timing_collector: Optional[TimingCollector] = None,
        comm: Optional[pace.util.Comm] = None,
    ):
        self._actual = FrozenStencil(
            func=func,
            origin=origin,
            domain=domain,
            stencil_config=stencil_config,
            externals=externals,
            skip_passes=skip_passes,
            timing_collector=timing_collector,
            comm=comm,
        )
        compilation_config = CompilationConfig(
            backend="numpy",
            rebuild=stencil_config.compilation_config.rebuild,
            validate_args=stencil_config.compilation_config.validate_args,
            format_source=True,
            device_sync=None,
            run_mode=RunMode.BuildAndRun,
            use_minimal_caching=False,
        )
        numpy_stencil_config = StencilConfig(
            dace_config=stencil_config.dace_config,
            compilation_config=compilation_config,
        )
        self._numpy = FrozenStencil(
            func=func,
            origin=origin,
            domain=domain,
            stencil_config=numpy_stencil_config,
            externals=externals,
            skip_passes=skip_passes,
            timing_collector=timing_collector,
            comm=comm,
        )
        self._func_name = func.__name__

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)
        self._actual(*args, **kwargs)
        self._numpy(*args_copy, **kwargs_copy)
        report_difference(
            args,
            kwargs,
            args_copy,
            kwargs_copy,
            self._func_name,
            self._actual.stencil_object._gt_id_,
        )


def _stencil_object_name(stencil_object: gt4py.StencilObject) -> str:
    """Returns a unique name for each gt4py stencil object, including the hash."""
    return type(stencil_object).__name__


def get_pair_rank(rank: int, size: int):
    dycore_ranks = size // 2
    if rank < dycore_ranks:
        return rank + dycore_ranks
    else:
        return rank - dycore_ranks


def compare_ranks(comm: pace.util.Comm, data) -> Mapping[str, int]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    pair_rank = get_pair_rank(rank, size)
    differences = {}
    for name, maybe_array in sorted(data.items(), key=lambda x: x[0]):
        if isinstance(maybe_array, pace.util.Quantity):
            maybe_array = maybe_array.data
        if hasattr(maybe_array, "data") and isinstance(maybe_array.data, np.ndarray):
            array = maybe_array.data
            other = comm.sendrecv(array, pair_rank)
            arr_diffs = np.sum(np.logical_and(~np.isnan(array), array != other))
            if arr_diffs > 0:
                print(name, rank, pair_rank, array, other)
                differences[name] = arr_diffs
    return differences


class FrozenStencil(SDFGConvertible):
    """
    Wrapper for gt4py stencils which stores origin and domain at compile time,
    and uses their stored values at call time.

    This is useful when the stencil itself is meant to be used on a certain
    grid, for example if a compile-time external variable is tied to the
    values of origin and domain.
    """

    def __init__(
        self,
        func: Callable[..., None],
        origin: Union[Tuple[int, ...], Mapping[str, Tuple[int, ...]]],
        domain: Tuple[int, ...],
        stencil_config: StencilConfig,
        externals: Optional[Mapping[str, Any]] = None,
        skip_passes: Tuple[str, ...] = (),
        timing_collector: Optional[TimingCollector] = None,
        comm: Optional[pace.util.Comm] = None,
    ):
        """
        Args:
            func: stencil definition function
            origin: gt4py origin to use at call time
            domain: gt4py domain to use at call time
            stencil_config: container for stencil configuration
            externals: compile-time external variables required by stencil
            skip_passes: compiler passes to skip when building stencil
            timing_collector: Optional object that accumulates timings
            comm: if given, inputs and outputs will be compared to the "twin"
                rank of this rank
        """
        if isinstance(origin, tuple):
            origin = cast_to_index3d(origin)
        origin = cast(Union[Index3D, Mapping[str, Tuple[int, ...]]], origin)
        self.origin = origin
        self.domain: Index3D = cast_to_index3d(domain)
        self.stencil_config: StencilConfig = stencil_config
        self.comm = comm

        if timing_collector is None:
            self._timing_collector = TimingCollector()
        else:
            self._timing_collector = timing_collector

        if externals is None:
            externals = {}
        self.externals = externals
        self._func_name = func.__name__
        stencil_kwargs = self.stencil_config.stencil_kwargs(
            skip_passes=skip_passes, func=func
        )
        self.stencil_object: Optional[gt4py.StencilObject] = None

        self._argument_names = tuple(inspect.getfullargspec(func).args)

        if "dace" in self.stencil_config.compilation_config.backend:
            dace.Config.set(
                "default_build_folder",
                value="{gt_cache}/dacecache".format(
                    gt_cache=gt4py.config.cache_settings["dir_name"]
                ),
            )

        assert (
            len(self._argument_names) > 0
        ), "A stencil with no arguments? You may be double decorating"

        # Keep compilation at __init__ if we are not orchestrated.
        # If we orchestrate, move the compilation at call time to make sure
        # disable_codegen do not lead to call to uncompiled stencils, which fails
        # silently
        if self.stencil_config.dace_config.is_dace_orchestrated():
            self.stencil_object = gtscript.lazy_stencil(
                definition=func,
                externals=externals,
                **stencil_kwargs,
                build_info=(build_info := {}),  # type: ignore
            )
        else:
            compilation_config = stencil_config.compilation_config
            if (
                compilation_config.use_minimal_caching
                and not compilation_config.is_compiling
                and compilation_config.run_mode != RunMode.Run
            ):
                block_waiting_for_compilation(MPI.COMM_WORLD, compilation_config)

            self.stencil_object = gtscript.stencil(
                definition=func,
                externals=externals,
                **stencil_kwargs,
                build_info=(build_info := {}),
            )

            if (
                compilation_config.use_minimal_caching
                and compilation_config.is_compiling
                and compilation_config.run_mode != RunMode.Run
            ):
                unblock_waiting_tiles(MPI.COMM_WORLD)

        self._timing_collector.build_info[
            _stencil_object_name(self.stencil_object)
        ] = build_info
        field_info = self.stencil_object.field_info

        self._field_origins: Dict[
            str, Tuple[int, ...]
        ] = FrozenStencil._compute_field_origins(field_info, self.origin)
        """mapping from field names to field origins"""

        self._stencil_run_kwargs: Dict[str, Any] = {
            "_origin_": self._field_origins,
            "_domain_": self.domain,
        }

        self._written_fields: List[str] = FrozenStencil._get_written_fields(field_info)

        if stencil_config.compilation_config.run_mode == RunMode.Build:

            def nothing_function(*args, **kwargs):
                pass

            setattr(self, "__call__", nothing_function)

    def __call__(self, *args, **kwargs) -> None:
        args_list = list(args)
        _convert_quantities_to_storage(args_list, kwargs)
        args = tuple(args_list)

        args_as_kwargs = dict(zip(self._argument_names, args))
        if self.comm is not None:
            differences = compare_ranks(self.comm, {**args_as_kwargs, **kwargs})
            if len(differences) > 0:
                raise ValueError(
                    f"rank {self.comm.Get_rank()} has differences {differences} "
                    f"before calling {self._func_name}"
                )
        if self.stencil_config.compilation_config.validate_args:
            if __debug__ and "origin" in kwargs:
                raise TypeError("origin cannot be passed to FrozenStencil call")
            if __debug__ and "domain" in kwargs:
                raise TypeError("domain cannot be passed to FrozenStencil call")
            self.stencil_object(
                *args,
                **kwargs,
                origin=self._field_origins,
                domain=self.domain,
                validate_args=True,
                exec_info=self._timing_collector.exec_info,
            )
        else:
            self.stencil_object.run(
                **args_as_kwargs,
                **kwargs,
                **self._stencil_run_kwargs,
                exec_info=self._timing_collector.exec_info,
            )
            self._mark_cuda_fields_written({**args_as_kwargs, **kwargs})
        if self.comm is not None:
            differences = compare_ranks(self.comm, {**args_as_kwargs, **kwargs})
            if len(differences) > 0:
                raise ValueError(
                    f"rank {self.comm.Get_rank()} has differences {differences} "
                    f"after calling {self._func_name}"
                )

    def _mark_cuda_fields_written(self, fields: Mapping[str, Storage]):
        if self.stencil_config.is_gpu_backend:
            for write_field in self._written_fields:
                fields[write_field]._set_device_modified()

    @classmethod
    def _compute_field_origins(
        cls, field_info_mapping, origin: Union[Index3D, Mapping[str, Tuple[int, ...]]]
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Computes the origin for each field in the stencil call.

        Args:
            field_info_mapping: from stencil.field_info, a mapping which gives the
                dimensionality of each input field
            origin: the (i, j, k) coordinate of the origin

        Returns:
            origin_mapping: a mapping from field names to origins
        """
        if isinstance(origin, tuple):
            field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
            origin_tuple: Tuple[int, ...] = origin
        else:
            field_origins = {**origin}
            origin_tuple = origin["_all_"]
        field_names = tuple(field_info_mapping.keys())
        for i, field_name in enumerate(field_names):
            if field_name not in field_origins:
                field_info = field_info_mapping[field_name]
                if field_info is not None:
                    field_origin_list = []
                    for ax in field_info.axes:
                        origin_index = {"I": 0, "J": 1, "K": 2}[ax]
                        field_origin_list.append(origin_tuple[origin_index])
                    field_origin = tuple(field_origin_list)
                else:
                    field_origin = origin_tuple
                field_origins[field_name] = field_origin
        return field_origins

    @classmethod
    def _get_written_fields(cls, field_info) -> List[str]:
        """Returns the list of fields that are written.

        Args:
            field_info: field_info attribute of gt4py stencil object
        """
        write_fields = [
            field_name
            for field_name in field_info
            if field_info[field_name]
            and bool(field_info[field_name].access & gt4py.definitions.AccessKind.WRITE)
        ]
        return write_fields

    @classmethod
    def _get_oir_pipeline(cls, skip_passes: Sequence[str]) -> OirPipeline:
        step_map = {step.__name__: step for step in DefaultPipeline.all_steps()}
        skip_steps = [step_map[pass_name] for pass_name in skip_passes]
        return DefaultPipeline(skip=skip_steps)

    def __sdfg__(self, *args, **kwargs):
        """Implemented SDFG generation"""
        args_as_kwargs = dict(zip(self._argument_names, args))
        return self.stencil_object.__sdfg__(
            origin=self._field_origins,
            domain=self.domain,
            **args_as_kwargs,
            **kwargs,
        )

    def __sdfg_signature__(self):
        """Implemented SDFG signature lookup"""
        return self.stencil_object.__sdfg_signature__()

    def __sdfg_closure__(self, *args, **kwargs):
        """Implemented SDFG closure build"""
        return self.stencil_object.__sdfg_closure__(*args, **kwargs)

    def closure_resolver(self, constant_args, given_args, parent_closure=None):
        """Implemented SDFG closure resolver build"""
        return self.stencil_object.closure_resolver(
            constant_args, given_args, parent_closure=parent_closure
        )


def _convert_quantities_to_storage(args, kwargs):
    for i, arg in enumerate(args):
        try:
            args[i] = arg.storage
        except AttributeError:
            pass
    for name, arg in kwargs.items():
        try:
            kwargs[name] = arg.storage
        except AttributeError:
            pass


class GridIndexing:
    """
    Provides indices for cell-centered variables with halos.

    These indices can be used with horizontal interface variables by adding 1
    to the domain shape along any interface axis.
    """

    def __init__(
        self,
        domain: Index3D,
        n_halo: int,
        south_edge: bool,
        north_edge: bool,
        west_edge: bool,
        east_edge: bool,
    ):
        """
        Initialize a grid indexing object.

        Args:
            domain: size of the compute domain for cell-centered variables
            n_halo: number of halo points
            south_edge: whether the current rank is on the south edge of a tile
            north_edge: whether the current rank is on the north edge of a tile
            west_edge: whether the current rank is on the west edge of a tile
            east_edge: whether the current rank is on the east edge of a tile
        """
        self.origin = (n_halo, n_halo, 0)
        self.n_halo = n_halo
        self.domain = domain
        self.south_edge = south_edge
        self.north_edge = north_edge
        self.west_edge = west_edge
        self.east_edge = east_edge

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain
        self._sizer = pace.util.SubtileGridSizer(
            nx=domain[0],
            ny=domain[1],
            nz=domain[2],
            n_halo=self.n_halo,
            extra_dim_lengths={},
        )

    @classmethod
    def from_sizer_and_communicator(
        cls, sizer: pace.util.GridSizer, cube: pace.util.CubedSphereCommunicator
    ) -> "GridIndexing":
        # TODO: if this class is refactored to split off the *_edge booleans,
        # this init routine can be refactored to require only a GridSizer
        domain = cast(
            Tuple[int, int, int],
            sizer.get_extent([pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]),
        )
        south_edge = cube.tile.partitioner.on_tile_bottom(cube.rank)
        north_edge = cube.tile.partitioner.on_tile_top(cube.rank)
        west_edge = cube.tile.partitioner.on_tile_left(cube.rank)
        east_edge = cube.tile.partitioner.on_tile_right(cube.rank)
        return cls(
            domain=domain,
            n_halo=sizer.n_halo,
            south_edge=south_edge,
            north_edge=north_edge,
            west_edge=west_edge,
            east_edge=east_edge,
        )

    @property
    def max_shape(self):
        """
        Maximum required storage shape, corresponding to the shape of a cell-corner
        variable with maximum halo points.

        This should rarely be required, consider using appropriate calls to helper
        methods that get the correct shape for your particular variable.
        """
        # need to add back origin as buffer points, what we're returning here
        # isn't a domain - it's an array size
        return self.domain_full(add=(1, 1, 1 + self.origin[2]))

    @property
    def isc(self):
        """start of the compute domain along the x-axis"""
        return self.origin[0]

    @property
    def iec(self):
        """last index of the compute domain along the x-axis"""
        return self.origin[0] + self.domain[0] - 1

    @property
    def jsc(self):
        """start of the compute domain along the y-axis"""
        return self.origin[1]

    @property
    def jec(self):
        """last index of the compute domain along the y-axis"""
        return self.origin[1] + self.domain[1] - 1

    @property
    def isd(self):
        """start of the full domain including halos along the x-axis"""
        return self.origin[0] - self.n_halo

    @property
    def ied(self):
        """index of the last data point along the x-axis"""
        return self.isd + self.domain[0] + 2 * self.n_halo - 1

    @property
    def jsd(self):
        """start of the full domain including halos along the y-axis"""
        return self.origin[1] - self.n_halo

    @property
    def jed(self):
        """index of the last data point along the y-axis"""
        return self.jsd + self.domain[1] + 2 * self.n_halo - 1

    @property
    def nw_corner(self):
        return self.north_edge and self.west_edge

    @property
    def sw_corner(self):
        return self.south_edge and self.west_edge

    @property
    def ne_corner(self):
        return self.north_edge and self.east_edge

    @property
    def se_corner(self):
        return self.south_edge and self.east_edge

    def origin_full(self, add: Index3D = (0, 0, 0)):
        """
        Returns the origin of the full domain including halos, plus an optional offset.
        """
        return (self.isd + add[0], self.jsd + add[1], self.origin[2] + add[2])

    def origin_compute(self, add: Index3D = (0, 0, 0)):
        """
        Returns the origin of the compute domain, plus an optional offset
        """
        return (self.isc + add[0], self.jsc + add[1], self.origin[2] + add[2])

    def domain_full(self, add: Index3D = (0, 0, 0)):
        """
        Returns the shape of the full domain including halos, plus an optional offset.
        """
        return (
            self.ied + 1 - self.isd + add[0],
            self.jed + 1 - self.jsd + add[1],
            self.domain[2] + add[2],
        )

    def domain_compute(self, add: Index3D = (0, 0, 0)):
        """
        Returns the shape of the compute domain, plus an optional offset.
        """
        return (
            self.iec + 1 - self.isc + add[0],
            self.jec + 1 - self.jsc + add[1],
            self.domain[2] + add[2],
        )

    def axis_offsets(
        self,
        origin: Tuple[int, ...],
        domain: Tuple[int, ...],
    ) -> Dict[str, Any]:
        if self.west_edge:
            i_start = gtscript.I[0] + self.origin[0] - origin[0]
        else:
            i_start = gtscript.I[0] - np.iinfo(np.int16).max

        if self.east_edge:
            i_end = (
                gtscript.I[-1]
                + (self.origin[0] + self.domain[0])
                - (origin[0] + domain[0])
            )
        else:
            i_end = gtscript.I[-1] + np.iinfo(np.int16).max

        if self.south_edge:
            j_start = gtscript.J[0] + self.origin[1] - origin[1]
        else:
            j_start = gtscript.J[0] - np.iinfo(np.int16).max

        if self.north_edge:
            j_end = (
                gtscript.J[-1]
                + (self.origin[1] + self.domain[1])
                - (origin[1] + domain[1])
            )
        else:
            j_end = gtscript.J[-1] + np.iinfo(np.int16).max

        return {
            "i_start": i_start,
            "local_is": gtscript.I[0] + self.isc - origin[0],
            "i_end": i_end,
            "local_ie": gtscript.I[-1] + self.iec - origin[0] - domain[0] + 1,
            "j_start": j_start,
            "local_js": gtscript.J[0] + self.jsc - origin[1],
            "j_end": j_end,
            "local_je": gtscript.J[-1] + self.jec - origin[1] - domain[1] + 1,
        }

    def get_origin_domain(
        self, dims: Sequence[str], halos: Sequence[int] = tuple()
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Get the origin and domain for a computation that occurs over a certain grid
        configuration (given by dims) and a certain number of halo points.

        Args:
            dims: dimension names, using dimension constants from pace.util
            halos: number of halo points for each dimension, defaults to zero

        Returns:
            origin: origin of the computation
            domain: shape of the computation
        """
        origin = self._origin_from_dims(dims)
        domain = list(self._sizer.get_extent(dims))
        for i, n in enumerate(halos):
            origin[i] -= n
            domain[i] += 2 * n
        return tuple(origin), tuple(domain)

    def _origin_from_dims(self, dims: Iterable[str]) -> List[int]:
        return_origin = []
        for dim in dims:
            if dim in pace.util.X_DIMS:
                return_origin.append(self.origin[0])
            elif dim in pace.util.Y_DIMS:
                return_origin.append(self.origin[1])
            elif dim in pace.util.Z_DIMS:
                return_origin.append(self.origin[2])
        return return_origin

    def get_shape(
        self, dims: Sequence[str], halos: Sequence[int] = tuple()
    ) -> Tuple[int, ...]:
        """
        Get the storage shape required for an array with the given dimensions
        which is accessed up to a given number of halo points.

        Args:
            dims: dimension names, using dimension constants from pace.util
            halos: number of halo points for each dimension, defaults to zero

        Returns:
            origin: origin of the computation
            domain: shape of the computation
        """
        shape = list(self._sizer.get_extent(dims))
        for i, d in enumerate(dims):
            # need n_halo points at the start of the domain, regardless of whether
            # they are read, so that data is aligned in memory
            if d in (pace.util.X_DIMS + pace.util.Y_DIMS):
                shape[i] += self.n_halo
        for i, n in enumerate(halos):
            shape[i] += n
        return tuple(shape)

    def restrict_vertical(self, k_start=0, nk=None) -> "GridIndexing":
        """
        Returns a copy of itself with modified vertical origin and domain.

        Args:
            k_start: offset to apply to current vertical origin, must be
                greater than 0 and less than the size of the vertical domain
            nk: new vertical domain size as a number of grid cells,
                defaults to remaining grid cells in the current domain,
                can be at most the size of the vertical domain minus k_start
        """
        if k_start < 0:
            raise ValueError("k_start must be positive")
        if k_start > self.domain[2]:
            raise ValueError(
                "k_start must be less than the number of vertical levels "
                f"(received {k_start} for {self.domain[2]} vertical levels"
            )
        if nk is None:
            nk = self.domain[2] - k_start
        elif nk < 0:
            raise ValueError("number of vertical levels should be positive")
        elif nk > (self.domain[2] - k_start):
            raise ValueError(
                "nk can be at most the size of the vertical domain minus k_start"
            )

        new = GridIndexing(
            self.domain[:2] + (nk,),
            self.n_halo,
            self.south_edge,
            self.north_edge,
            self.west_edge,
            self.east_edge,
        )
        new.origin = self.origin[:2] + (self.origin[2] + k_start,)
        return new

    def get_quantity_halo_spec(
        self,
        shape: Tuple[int, ...],
        origin: Tuple[int, ...],
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        n_halo: Optional[int] = None,
        *,
        backend: str,
    ) -> QuantityHaloSpec:
        """Build memory specifications for the halo update.

        Args:
            shape: the shape of the Quantity
            origin: the origin of the compute domain
            dims: dimensionality of the data
            n_halo: number of halo points to update, defaults to self.n_halo
            backend: gt4py backend to use
        """

        # TEMPORARY: we do a nasty temporary allocation here to read in the hardware
        # memory layout. Further work in GT4PY will allow for deferred allocation
        # which will give access to those information while making sure
        # we don't allocate
        # Refactor is filed in ticket DSL-820

        temp_storage = gt4py_utils.make_storage_from_shape(
            shape, origin, backend=backend
        )
        origin, extent = self.get_origin_domain(dims)
        temp_quantity = pace.util.Quantity(
            temp_storage,
            dims=dims,
            units="unknown",
            origin=origin,
            extent=extent,
        )
        if n_halo is None:
            n_halo = self.n_halo

        spec = QuantityHaloSpec(
            n_halo,
            temp_quantity.data.strides,
            temp_quantity.data.itemsize,
            temp_quantity.data.shape,
            temp_quantity.metadata.origin,
            temp_quantity.metadata.extent,
            temp_quantity.metadata.dims,
            temp_quantity.np,
            temp_quantity.metadata.dtype,
        )

        del temp_storage
        del temp_quantity

        return spec


class StencilFactory:
    """Configurable class which creates stencil objects."""

    def __init__(
        self,
        config: StencilConfig,
        grid_indexing: GridIndexing,
        comm: Optional[pace.util.Comm] = None,
    ):
        """
        Args:
            config: gt4py-specific stencil configuration
            grid_indexing: configuration for domain and halo indexing
            comm: if given, stencils will compare all data before and after
                stencil execution to their "pair" rank on the comm. This is very
                expensive and only used for debugging.
        """
        self.config: StencilConfig = config
        self.grid_indexing: GridIndexing = grid_indexing
        self.timing_collector = TimingCollector()
        self.comm = comm

    @property
    def backend(self):
        return self.config.compilation_config.backend

    def from_origin_domain(
        self,
        func: Callable[..., None],
        origin: Union[Tuple[int, ...], Mapping[str, Tuple[int, ...]]],
        domain: Tuple[int, ...],
        externals: Optional[Mapping[str, Any]] = None,
        skip_passes: Tuple[str, ...] = (),
    ) -> Union[FrozenStencil, CompareToNumpyStencil]:
        """
        Args:
            func: stencil definition function
            origin: gt4py origin to use at call time
            domain: gt4py domain to use at call time
            stencil_config: container for stencil configuration
            externals: compile-time external variables required by stencil
            skip_passes: compiler passes to skip when building stencil
        """
        if self.config.compare_to_numpy:
            cls: Type = CompareToNumpyStencil
        else:
            cls = FrozenStencil
        return cls(
            func=func,
            origin=origin,
            domain=domain,
            stencil_config=self.config,
            externals=externals,
            skip_passes=skip_passes,
            timing_collector=self.timing_collector,
            comm=self.comm,
        )

    def from_dims_halo(
        self,
        func: Callable[..., None],
        compute_dims: Sequence[str],
        compute_halos: Sequence[int] = tuple(),
        externals: Optional[Mapping[str, Any]] = None,
        skip_passes: Tuple[str, ...] = (),
    ) -> Union[FrozenStencil, CompareToNumpyStencil]:
        """
        Initialize a stencil from dimensions and number of halo points.

        Automatically injects axis_offsets into stencil externals.

        Args:
            func: stencil definition function
            compute_dims: dimensionality of compute domain
            compute_halos: number of halo points to include in compute domain
            externals: compile-time external variables required by stencil
            skip_passes: compiler passes to skip when building stencil
        """
        if externals is None:
            externals = {}
        if len(compute_dims) != 3:
            raise ValueError(
                f"must have 3 dimensions to create stencil, got {compute_dims}"
            )
        origin, domain = self.grid_indexing.get_origin_domain(
            dims=compute_dims, halos=compute_halos
        )
        origin = cast_to_index3d(origin)
        domain = cast_to_index3d(domain)
        all_externals = self.grid_indexing.axis_offsets(origin=origin, domain=domain)
        all_externals.update(externals)
        return self.from_origin_domain(
            func=func,
            origin=origin,
            domain=domain,
            externals=all_externals,
            skip_passes=skip_passes,
        )

    def restrict_vertical(self, k_start=0, nk=None) -> "StencilFactory":
        return StencilFactory(
            config=self.config,
            grid_indexing=self.grid_indexing.restrict_vertical(k_start=k_start, nk=nk),
            comm=self.comm,
        )

    def build_report(self, key: str = "build_time", **kwargs) -> str:
        """Report all stencils built by this factory."""
        return self.timing_collector.build_report(key, **kwargs)

    def exec_report(self, key: str = "total_run_time", **kwargs) -> str:
        """Report all stencils executed that were built by this factory."""
        return self.timing_collector.exec_report(key, **kwargs)


def get_stencils_with_varied_bounds(
    func: Callable[..., None],
    origins: List[Index3D],
    domains: List[Index3D],
    stencil_factory: StencilFactory,
    externals: Optional[Mapping[str, Any]] = None,
) -> List[Union[FrozenStencil, CompareToNumpyStencil]]:
    assert len(origins) == len(domains), (
        "Lists of origins and domains need to have the same length, you provided "
        + str(len(origins))
        + " origins and "
        + str(len(domains))
        + " domains"
    )
    if externals is None:
        externals = {}
    stencils = []
    for origin, domain in zip(origins, domains):
        ax_offsets = stencil_factory.grid_indexing.axis_offsets(
            origin=origin, domain=domain
        )
        stencils.append(
            stencil_factory.from_origin_domain(
                func,
                origin=origin,
                domain=domain,
                externals={**externals, **ax_offsets},
            )
        )
    return stencils
