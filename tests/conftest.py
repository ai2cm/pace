import collections
import os
import sys
import warnings

import pytest
import translate
import yaml

import fv3core
import fv3core._config
import fv3core.testing
import fv3core.utils.gt4py_utils
import fv3gfs.util as fv3util
from fv3core.testing import ParallelTranslate, TranslateGrid
from fv3core.utils.mpi import MPI


# get MPI environment
sys.path.append("/usr/local/serialbox/python")  # noqa: E402
import serialbox  # noqa: E402


GRID_SAVEPOINT_NAME = "Grid-Info"

# this must happen before any classes from fv3core are instantiated
fv3core.testing.enable_selective_validation()


class ReplaceRepr:
    def __init__(self, wrapped, new_repr):
        self._wrapped = wrapped
        self._repr = new_repr

    def __repr__(self):
        return self._repr

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    fv3core.set_backend(backend)
    return backend


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_from_config(pytestconfig)


def data_path_from_config(config):
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(data_path, "input.nml")
    fv3core._config.set_namelist(namelist_filename)
    return data_path


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


def make_grid(grid_savepoint, serializer, rank):
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = read_serialized_data(serializer, grid_savepoint, field)
    return TranslateGrid(grid_data, rank).python_grid()


def read_serialized_data(serializer, savepoint, variable):

    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def process_grid_savepoint(serializer, grid_savepoint, rank):
    grid = make_grid(grid_savepoint, serializer, rank)
    fv3core._config.set_grid(grid)
    return grid


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


def get_test_class_instance(test_name, grid):
    translate_class = get_test_class(test_name)
    if translate_class is None:
        return None
    else:
        return translate_class(grid)


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
    ],
)


def sequential_savepoint_cases(metafunc, data_path):
    return_list = []
    layout = fv3core._config.namelist.layout
    total_ranks = 6 * layout[0] * layout[1]
    savepoint_names = get_sequential_savepoint_names(metafunc, data_path)

    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
        grid = process_grid_savepoint(serializer, grid_savepoint, rank)
        if rank == 0:
            grid_rank0 = grid
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
                    layout,
                )
            )
    fv3core._config.set_grid(grid_rank0)
    return return_list


def check_savepoint_counts(test_name, input_savepoints, output_savepoints):
    if len(input_savepoints) != len(output_savepoints):
        warnings.warn(
            f"number of input and output savepoints not equal for {test_name}:"
            f" {len(input_savepoints)} in and {len(output_savepoints)} out"
        )
    assert len(input_savepoints) > 0, f"no savepoints found for {test_name}"


def mock_parallel_savepoint_cases(metafunc, data_path):
    return_list = []
    layout = fv3core._config.namelist.layout
    total_ranks = 6 * layout[0] * layout[1]
    grid_list = []
    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
        grid = process_grid_savepoint(serializer, grid_savepoint, rank)
        grid_list.append(grid)
        if rank == 0:
            grid_rank0 = grid
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
                layout,
            )
        )
    fv3core._config.set_grid(grid_rank0)
    return return_list


def parallel_savepoint_cases(metafunc, data_path, mpi_rank):
    serializer = get_serializer(data_path, mpi_rank)
    grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
    grid = process_grid_savepoint(serializer, grid_savepoint, mpi_rank)
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    return_list = []
    layout = fv3core._config.namelist.layout
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
                layout,
            )
        )
    return return_list


def pytest_generate_tests(metafunc):
    backend = metafunc.config.getoption("backend")
    fv3core.set_backend(backend)
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        if metafunc.function.__name__ == "test_parallel_savepoint":
            generate_parallel_stencil_tests(metafunc)
    else:
        if metafunc.function.__name__ == "test_sequential_savepoint":
            generate_sequential_stencil_tests(metafunc)
        if metafunc.function.__name__ == "test_mock_parallel_savepoint":
            generate_mock_parallel_stencil_tests(metafunc)


def generate_sequential_stencil_tests(metafunc):
    arg_names = [
        "testobj",
        "test_name",
        "serializer",
        "savepoint_in",
        "savepoint_out",
        "rank",
        "grid",
    ]
    data_path = data_path_from_config(metafunc.config)
    _generate_stencil_tests(
        metafunc,
        arg_names,
        sequential_savepoint_cases(metafunc, data_path),
        get_sequential_param,
    )


def generate_mock_parallel_stencil_tests(metafunc):
    arg_names = [
        "testobj",
        "test_name",
        "serializer_list",
        "savepoint_in_list",
        "savepoint_out_list",
        "grid",
        "layout",
    ]
    data_path = data_path_from_config(metafunc.config)
    _generate_stencil_tests(
        metafunc,
        arg_names,
        mock_parallel_savepoint_cases(metafunc, data_path),
        get_parallel_mock_param,
    )


def generate_parallel_stencil_tests(metafunc):
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
    data_path = data_path_from_config(metafunc.config)
    # get MPI environment
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    _generate_stencil_tests(
        metafunc,
        arg_names,
        parallel_savepoint_cases(metafunc, data_path, mpi_rank),
        get_parallel_param,
    )


def _generate_stencil_tests(metafunc, arg_names, savepoint_cases, get_param):
    param_list = []
    for case in savepoint_cases:
        original_grid = fv3core._config.grid
        try:
            fv3core._config.set_grid(case.grid)
            testobj = get_test_class_instance(case.test_name, case.grid)
        finally:
            fv3core._config.set_grid(original_grid)
        max_call_count = min(len(case.input_savepoints), len(case.output_savepoints))
        for i, (savepoint_in, savepoint_out) in enumerate(
            zip(case.input_savepoints, case.output_savepoints)
        ):
            param_list.append(
                get_param(case, testobj, savepoint_in, savepoint_out, i, max_call_count)
            )

    metafunc.parametrize(", ".join(arg_names), param_list)


def get_parallel_param(
    case, testobj, savepoint_in, savepoint_out, call_count, max_call_count
):
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
    case, testobj, savepoint_in_list, savepoint_out_list, call_count, max_call_count
):
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
    case, testobj, savepoint_in, savepoint_out, call_count, max_call_count
):
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
        marks=pytest.mark.dependency(
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
        ),
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


def pytest_addoption(parser):
    parser.addoption("--which_modules", action="store")
    parser.addoption("--skip_modules", action="store")
    parser.addoption("--print_failures", action="store_true")
    parser.addoption("--failure_stride", action="store", default=1)
    parser.addoption("--data_path", action="store", default="./")
    parser.addoption("--backend", action="store", default="numpy")
    parser.addoption("--python_regression", action="store_true")
    parser.addoption("--threshold_overrides_file", action="store", default=None)


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "sequential(name): mark test as running sequentially on ranks"
    )
    config.addinivalue_line(
        "markers", "parallel(name): mark test as running in parallel across ranks"
    )
    config.addinivalue_line(
        "markers",
        "mock_parallel(name): mark test as running in mock parallel across ranks",
    )


@pytest.fixture()
def print_failures(pytestconfig):
    return pytestconfig.getoption("print_failures")


@pytest.fixture()
def failure_stride(pytestconfig):
    return int(pytestconfig.getoption("failure_stride"))


@pytest.fixture()
def python_regression(pytestconfig):
    return pytestconfig.getoption("python_regression")
