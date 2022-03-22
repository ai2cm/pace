import collections
import os
import sys
import warnings
from typing import Tuple

import f90nml
import pytest
import yaml

import pace.dsl
import pace.util as fv3util
from pace.stencils.testing.parallel_translate import ParallelTranslate
from pace.stencils.testing.translate import TranslateGrid
from pace.util.mpi import MPI
from pace.util.namelist import Namelist

from . import translate


# get MPI environment
sys.path.append("/usr/local/serialbox/python")  # noqa: E402
import serialbox  # noqa: E402


class ReplaceRepr:
    def __init__(self, wrapped, new_repr):
        self._wrapped = wrapped
        self._repr = new_repr

    def __repr__(self):
        return self._repr

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_and_namelist_filename_from_config(pytestconfig)


def data_path_and_namelist_filename_from_config(config) -> Tuple[str, str]:
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(data_path, "input.nml")
    return data_path, namelist_filename


@pytest.fixture
def threshold_overrides(pytestconfig):
    return thresholds_from_file(pytestconfig)


def thresholds_from_file(config):
    thresholds_file = config.getoption("threshold_overrides_file")
    if thresholds_file is None:
        return None
    return yaml.safe_load(open(thresholds_file, "r"))


@pytest.fixture
def serializer(data_path, rank):
    return get_serializer(data_path, rank)


def get_serializer(data_path, rank):
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
    )


def is_input_name(savepoint_name):
    return savepoint_name[-3:] == "-In"


def to_output_name(savepoint_name):
    return savepoint_name[-3:] + "-Out"


def read_serialized_data(serializer, savepoint, variable):

    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


@pytest.fixture
def stencil_config(backend):
    return pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )


def get_test_class(test_name):
    translate_class_name = f"Translate{test_name.replace('-', '_')}"
    try:
        return_class = getattr(translate, translate_class_name)
    except AttributeError as err:
        if translate_class_name in err.args[0]:
            return_class = None
        else:
            raise err
    return return_class


def is_parallel_test(test_name):
    test_class = get_test_class(test_name)
    if test_class is None:
        return False
    else:
        return issubclass(test_class, ParallelTranslate)


def get_test_class_instance(test_name, grid, namelist, stencil_factory):
    translate_class = get_test_class(test_name)
    if translate_class is None:
        return None
    else:
        return translate_class(grid, namelist, stencil_factory)


def get_all_savepoint_names(metafunc, data_path):
    only_names = metafunc.config.getoption("which_modules")
    if only_names is None:
        savepoint_names = set()
        serializer = get_serializer(data_path, rank=0)
        for savepoint in serializer.savepoint_list():
            if is_input_name(savepoint.name):
                savepoint_names.add(savepoint.name[:-3])
    else:
        savepoint_names = set(only_names.split(","))
        savepoint_names.discard("")
    skip_names = metafunc.config.getoption("skip_modules")
    if skip_names is not None:
        savepoint_names.difference_update(skip_names.split(","))
    return savepoint_names


def get_sequential_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    sequential_names = []
    for name in all_names:
        if not is_parallel_test(name):
            sequential_names.append(name)
    return sequential_names


def get_parallel_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    parallel_names = []
    for name in all_names:
        if is_parallel_test(name):
            parallel_names.append(name)
    return parallel_names


def get_ranks(metafunc, layout):
    only_rank = metafunc.config.getoption("which_rank")
    if only_rank is None:
        total_ranks = 6 * layout[0] * layout[1]
        return range(total_ranks)
    else:
        return [int(only_rank)]


SavepointCase = collections.namedtuple(
    "SavepointCase",
    [
        "test_name",
        "rank",
        "serializer",
        "input_savepoints",
        "output_savepoints",
        "grid",
        "layout",
        "namelist",
        "stencil_factory",
    ],
)


def sequential_savepoint_cases(
    metafunc,
    data_path,
    namelist_filename,
    *,
    backend: str,
):
    return_list = []
    namelist = Namelist.from_f90nml(f90nml.read(namelist_filename))

    savepoint_names = get_sequential_savepoint_names(metafunc, data_path)
    ranks = get_ranks(metafunc, namelist.layout)
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    for rank in ranks:
        serializer = get_serializer(data_path, rank)
        grid = TranslateGrid.new_from_serialized_data(
            serializer, rank, namelist.layout, backend
        ).python_grid()
        stencil_factory = pace.dsl.stencil.StencilFactory(
            config=stencil_config,
            grid_indexing=grid.grid_indexing,
        )
        for test_name in sorted(list(savepoint_names)):
            input_savepoints = serializer.get_savepoint(f"{test_name}-In")
            output_savepoints = serializer.get_savepoint(f"{test_name}-Out")
            check_savepoint_counts(test_name, input_savepoints, output_savepoints)
            return_list.append(
                SavepointCase(
                    test_name,
                    rank,
                    serializer,
                    input_savepoints,
                    output_savepoints,
                    grid,
                    namelist.layout,
                    namelist,
                    stencil_factory,
                )
            )
    return return_list


def check_savepoint_counts(test_name, input_savepoints, output_savepoints):
    if len(input_savepoints) != len(output_savepoints):
        warnings.warn(
            f"number of input and output savepoints not equal for {test_name}:"
            f" {len(input_savepoints)} in and {len(output_savepoints)} out"
        )
    assert len(input_savepoints) > 0, f"no savepoints found for {test_name}"


def mock_parallel_savepoint_cases(
    metafunc, data_path, namelist_filename, *, backend: str
):
    return_list = []
    namelist = Namelist.from_f90nml(f90nml.read(namelist_filename))
    total_ranks = 6 * namelist.layout[0] * namelist.layout[1]
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    grid_list = []
    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        grid = TranslateGrid.new_from_serialized_data(
            serializer, rank, namelist.layout, backend
        ).python_grid()
        grid_list.append(grid)
    stencil_factory = pace.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid.grid_indexing,
    )
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    for test_name in sorted(list(savepoint_names)):
        input_list = []
        output_list = []
        serializer_list = []
        for rank in range(total_ranks):
            serializer = get_serializer(data_path, rank)
            serializer_list.append(serializer)
            input_savepoints = serializer.get_savepoint(f"{test_name}-In")
            output_savepoints = serializer.get_savepoint(f"{test_name}-Out")
            check_savepoint_counts(test_name, input_savepoints, output_savepoints)
            input_list.append(input_savepoints)
            output_list.append(output_savepoints)
        return_list.append(
            SavepointCase(
                test_name,
                None,
                serializer_list,
                list(
                    zip(*input_list)
                ),  # input_list[rank][count] -> input_list[count][rank]
                list(zip(*output_list)),
                grid_list,
                namelist.layout,
                namelist,
                stencil_factory,
            )
        )
    return return_list


def parallel_savepoint_cases(
    metafunc, data_path, namelist_filename, mpi_rank, *, backend: str
):
    serializer = get_serializer(data_path, mpi_rank)
    namelist = Namelist.from_f90nml(f90nml.read(namelist_filename))
    stencil_config = pace.dsl.stencil.StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=True,
    )
    grid = TranslateGrid.new_from_serialized_data(
        serializer, mpi_rank, namelist.layout, backend
    ).python_grid()
    stencil_factory = pace.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid.grid_indexing,
    )
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    return_list = []
    for test_name in sorted(list(savepoint_names)):
        input_savepoints = serializer.get_savepoint(f"{test_name}-In")
        output_savepoints = serializer.get_savepoint(f"{test_name}-Out")
        check_savepoint_counts(test_name, input_savepoints, output_savepoints)
        return_list.append(
            SavepointCase(
                test_name,
                mpi_rank,
                serializer,
                input_savepoints,
                output_savepoints,
                [grid],
                namelist.layout,
                namelist,
                stencil_factory,
            )
        )
    return return_list


def pytest_generate_tests(metafunc):
    backend = metafunc.config.getoption("backend")
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        if metafunc.function.__name__ == "test_parallel_savepoint":
            generate_parallel_stencil_tests(metafunc, backend=backend)
    else:
        if metafunc.function.__name__ == "test_sequential_savepoint":
            generate_sequential_stencil_tests(metafunc, backend=backend)
        if metafunc.function.__name__ == "test_mock_parallel_savepoint":
            generate_mock_parallel_stencil_tests(metafunc, backend=backend)


def generate_sequential_stencil_tests(metafunc, *, backend: str):
    arg_names = [
        "testobj",
        "test_name",
        "serializer",
        "savepoint_in",
        "savepoint_out",
        "rank",
        "grid",
    ]
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    _generate_stencil_tests(
        metafunc,
        arg_names,
        sequential_savepoint_cases(
            metafunc,
            data_path,
            namelist_filename,
            backend=backend,
        ),
        get_sequential_param,
    )


def generate_mock_parallel_stencil_tests(metafunc, backend: str):
    arg_names = [
        "testobj",
        "test_name",
        "serializer_list",
        "savepoint_in_list",
        "savepoint_out_list",
        "grid",
        "layout",
    ]
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    _generate_stencil_tests(
        metafunc,
        arg_names,
        mock_parallel_savepoint_cases(
            metafunc, data_path, namelist_filename, backend=backend
        ),
        get_parallel_mock_param,
    )


def generate_parallel_stencil_tests(metafunc, backend: str):
    arg_names = [
        "testobj",
        "test_name",
        "test_case",
        "serializer",
        "savepoint_in",
        "savepoint_out",
        "grid",
        "layout",
    ]
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    # get MPI environment
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    _generate_stencil_tests(
        metafunc,
        arg_names,
        parallel_savepoint_cases(
            metafunc, data_path, namelist_filename, mpi_rank, backend=backend
        ),
        get_parallel_param,
    )


def _generate_stencil_tests(metafunc, arg_names, savepoint_cases, get_param):
    param_list = []
    only_one_rank = metafunc.config.getoption("which_rank") is not None
    for case in savepoint_cases:
        testobj = get_test_class_instance(
            case.test_name, case.grid, case.namelist, case.stencil_factory
        )
        max_call_count = min(len(case.input_savepoints), len(case.output_savepoints))
        for i, (savepoint_in, savepoint_out) in enumerate(
            zip(case.input_savepoints, case.output_savepoints)
        ):
            param_list.append(
                get_param(
                    case,
                    testobj,
                    savepoint_in,
                    savepoint_out,
                    i,
                    max_call_count,
                    only_one_rank,
                )
            )

    metafunc.parametrize(", ".join(arg_names), param_list)


def get_parallel_param(
    case,
    testobj,
    savepoint_in,
    savepoint_out,
    call_count,
    max_call_count,
    only_one_rank,
):
    if only_one_rank is not False:
        raise RuntimeError("Cannot use --which_rank with parallel tests.")

    test_case = f"{case.test_name}-rank={case.rank}--call_count={call_count}"
    return pytest.param(
        testobj,
        case.test_name,
        test_case,
        ReplaceRepr(case.serializer, f"<Serializer for rank {case.rank}>"),
        savepoint_in,
        savepoint_out,
        case.grid,
        case.layout,
        id=test_case,
    )


def get_parallel_mock_param(
    case,
    testobj,
    savepoint_in_list,
    savepoint_out_list,
    call_count,
    max_call_count,
    only_one_rank,
):
    if only_one_rank is not False:
        raise RuntimeError("Cannot use --which_rank with parallel tests.")

    return pytest.param(
        testobj,
        case.test_name,
        [
            ReplaceRepr(ser, f"<Serializer for rank {rank}>")
            for rank, ser in enumerate(case.serializer)
        ],
        savepoint_in_list,
        savepoint_out_list,
        case.grid,
        case.layout,
        id=f"{case.test_name}-call_count={call_count}",
        marks=pytest.mark.dependency(
            name=f"{case.test_name}-{call_count}",
            depends=[
                f"{case.test_name}-{lower_count}"
                for lower_count in range(0, call_count)
            ],
        ),
    )


def get_sequential_param(
    case,
    testobj,
    savepoint_in,
    savepoint_out,
    call_count,
    max_call_count,
    only_one_rank,
):
    dependency = (
        pytest.mark.dependency()
        if only_one_rank
        else pytest.mark.dependency(
            name=f"{case.test_name}-{case.rank}-{call_count}",
            depends=[
                f"{case.test_name}-{lower_rank}-{count}"
                for lower_rank in range(0, case.rank)
                for count in range(0, max_call_count)
            ]
            + [
                f"{case.test_name}-{case.rank}-{lower_count}"
                for lower_count in range(0, call_count)
            ],
        )
    )
    return pytest.param(
        testobj,
        case.test_name,
        # serializer repr is very verbose, and not all that useful, so we hide it here
        ReplaceRepr(case.serializer, f"<Serializer for rank {case.rank}>"),
        savepoint_in,
        savepoint_out,
        case.rank,
        case.grid,
        id=f"{case.test_name}-rank={case.rank}-call_count={call_count}",
        marks=dependency,
    )


@pytest.fixture()
def communicator(layout):
    communicator = get_communicator(MPI.COMM_WORLD, layout)
    return communicator


@pytest.fixture()
def mock_communicator_list(layout):
    return get_mock_communicator_list(layout)


def get_mock_communicator_list(layout):
    total_ranks = 6 * fv3util.TilePartitioner(layout).total_ranks
    shared_buffer = {}
    communicators = []
    for rank in range(total_ranks):
        comm = fv3util.testing.DummyComm(rank, total_ranks, buffer_dict=shared_buffer)
        communicator = get_communicator(comm, layout)
        communicators.append(communicator)
    return communicators


def get_communicator(comm, layout):
    partitioner = fv3util.CubedSpherePartitioner(fv3util.TilePartitioner(layout))
    communicator = fv3util.CubedSphereCommunicator(comm, partitioner)
    return communicator


@pytest.fixture()
def print_failures(pytestconfig):
    return pytestconfig.getoption("print_failures")


@pytest.fixture()
def failure_stride(pytestconfig):
    return int(pytestconfig.getoption("failure_stride"))


@pytest.fixture()
def print_domains(pytestconfig):
    value = bool(pytestconfig.getoption("print_domains"))
    original_init = pace.dsl.stencil.FrozenStencil.__init__
    try:
        if value:

            def __init__(self, func, origin, domain, *args, **kwargs):
                print(func.__name__, origin, domain)
                original_init(self, func, origin, domain, *args, **kwargs)

            pace.dsl.stencil.FrozenStencil.__init__ = __init__
        yield value
    finally:
        pace.dsl.stencil.FrozenStencil.__init__ = original_init


@pytest.fixture()
def python_regression(pytestconfig):
    return pytestconfig.getoption("python_regression")
