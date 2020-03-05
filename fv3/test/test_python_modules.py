#!/usr/bin/env python3

import os
import sys

sys.path.append("/serialbox2/install/python")  # noqa

import numpy as np
import serialbox as ser
from fv3.translate.translate import TranslateGrid
from pydoc import locate
import pytest
import fv3.utils.gt4py_utils as utils
import traceback
from .delayed_assert import expect, assert_expectations


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_test_class_instance(test_name, split_name, grid):

    if len(split_name) > 2:
        end_index = -2
    else:
        end_index = -1
    package_name = "-".join(split_name[0:end_index])
    test_class_name = (
        "fv3.translate.translate_"
        + package_name.lower()
        + ".Translate"
        + test_name.replace("-", "_")
    )

    print("testing", test_class_name)
    test_class = locate(test_class_name)
    if test_class is None:
        return None
    return test_class(grid)


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def check_savepoints_defined(sp_grid):
    if sp_grid is None:
        raise Exception("savepoint encountered before grid defined")


def check_isready(test_name, isready, expected_ready):
    if bool(test_name in isready) != bool(expected_ready):
        raise Exception("Out-of order data encountered!")


def collect_input_data(testobj, serializer, sp, sp_grid):
    input_data = {}
    for k in (
        testobj.serialnames(testobj.in_vars["data_vars"])
        + testobj.in_vars["parameters"]
    ):
        input_data[k] = read_serialized_data(serializer, sp, k)

    return input_data


def compare_arr(computed_data, ref_data):
    denom = np.abs(ref_data) + np.abs(computed_data)
    compare = 2.0 * np.abs(computed_data - ref_data) / denom
    compare[denom == 0] = 0.0
    # compare = np.abs((computed_data - ref_data)) / np.abs(ref_data)
    # compare[ref_data == 0] = np.abs(computed_data[ref_data == 0] - ref_data[ref_data == 0])
    return compare


def success_array(computed_data, ref_data, eps):
    return np.logical_or(
        compare_arr(computed_data, ref_data) < eps,
        np.logical_and(np.isnan(computed_data), np.isnan(ref_data)),
    )


def success(computed_data, ref_data, eps):
    return np.all(success_array(computed_data, ref_data, eps))


def sample_wherefail(
    computed_data, ref_data, eps, print_failures, failure_stride, test_str=""
):
    found_indices = np.where(
        np.logical_not(success_array(computed_data, ref_data, eps))
    )

    bad_indices_count = len(found_indices[0])
    if print_failures:
        bad_count = 0
        for b in range(bad_indices_count):
            if bad_count % failure_stride == 0:
                full_index = [f[b] for f in found_indices]
                print(
                    test_str,
                    "index:",
                    full_index,
                    computed_data[found_indices][b],
                    ref_data[found_indices][b],
                    abs(computed_data[found_indices][b] - ref_data[found_indices][b]),
                )
            bad_count += 1
    sample = [f[0] for f in found_indices]
    fullcount = len(ref_data.flatten())
    return (
        "Failed count: "
        + str(bad_indices_count)
        + "/"
        + str(fullcount)
        + "("
        + str(round(100.0 * (bad_indices_count / fullcount), 2))
        + "%), First failed index"
        + str(sample)
        + ", Computed value: "
        + str(computed_data[found_indices][0])
        + ", Reference serialized value: "
        + str(ref_data[found_indices][0])
        + ", Diff:"
        + str(abs(computed_data[found_indices][0] - ref_data[found_indices][0]))
    )


def make_grid(sp_grid, serializer):
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(sp_grid)
    for field in grid_fields:
        grid_data[field] = read_serialized_data(serializer, sp_grid, field)
    return TranslateGrid(grid_data).python_grid()


def savepoint_unique_name(sp):
    return sp.name + "-" + str(sp.metainfo["ID"])


@pytest.fixture()
def backend(pytestconfig):
    return pytestconfig.getoption("backend")


@pytest.fixture()
def data_backend(pytestconfig):
    return pytestconfig.getoption("data_backend")


@pytest.fixture()
def exec_backend(pytestconfig):
    return pytestconfig.getoption("exec_backend")


@pytest.fixture()
def which_modules(pytestconfig):
    return pytestconfig.getoption("which_modules").split(",")


@pytest.fixture()
def skip_modules(pytestconfig):
    skips = pytestconfig.getoption("skip_modules").split(",")
    skips = [skip for skip in skips if skip != "none"]
    return skips


@pytest.fixture()
def print_failures(pytestconfig):
    return pytestconfig.getoption("print_failures")


@pytest.fixture()
def failure_stride(pytestconfig):
    return int(pytestconfig.getoption("failure_stride"))


@pytest.fixture()
def data_path(pytestconfig):
    os.environ["NAMELIST_FILENAME"] = os.path.join(
        pytestconfig.getoption("data_path"), "input.nml"
    )
    return pytestconfig.getoption("data_path")


def interpret_success(test_name, var, ref_data, testobj, test_str, args):
    if success(args["output_data"][test_name][var], ref_data, testobj.max_error):
        print(bcolors.OKGREEN + "PASSED " + bcolors.ENDC + test_str)
        if success(
            args["output_data"][test_name][var] * 1.0001, ref_data, testobj.max_error
        ):
            print(
                bcolors.FAIL
                + "WARNING, this may not be a good test, possibly you do not have representative serialized data"
                + bcolors.ENDC,
                test_str,
                "data mean:",
                ref_data.mean(),
                "max error:",
                testobj.max_error,
            )
            del args["output_data"][test_name]
    else:
        print(
            var,
            args["output_data"][test_name][var].shape,
            ref_data.shape,
            testobj.max_error,
        )
        example = sample_wherefail(
            args["output_data"][test_name][var],
            ref_data,
            testobj.max_error,
            args["print_failures"],
            args["failure_stride"],
            test_str,
        )
        expect(False, msg="FAILED for " + test_str + "\n\t" + example)


def process_input_savepoint(serializer, sp, testobj, test_name, args):
    # check this is an appropriate time for a savepoint and that grid savepoint has already been encountered
    check_isready(test_name, args["isready"], False)
    check_savepoints_defined(args["sp_grid"])
    # read serialized stencil inputs
    input_data = collect_input_data(testobj, serializer, sp, args["sp_grid"])
    # run python version of functionality
    args["output_data"][test_name] = testobj.compute(input_data)
    if args["output_data"][test_name] is None:
        raise Exception(test_name + " did not return an array, it returned None")
    # Mark this test as ready to compare to serialized reference data
    args["isready"].append(test_name)


def process_output_savepoint(serializer, sp, testobj, test_name, args):
    check_isready(test_name, args["isready"], True)
    print("> Checking Savepoint", sp)
    for var in testobj.serialnames(testobj.out_vars):
        test_str = (
            "rank: "
            + str(args["rank"])
            + ", savepoint: "
            + sp.name
            + ", ID: "
            + str(sp.metainfo["ID"])
            + ", variable: "
            + var
        )
        ref_data = read_serialized_data(serializer, sp, var)
        interpret_success(test_name, var, ref_data, testobj, test_str, args)
    args["isready"].remove(test_name)


def process_test_savepoint(serializer, sp, split_name, args):
    test_name = "-".join(split_name[0:-1])
    if (
        args["which_modules"] != ["all"] and test_name not in args["which_modules"]
    ) or test_name in args["skip_modules"]:
        return 0
    # Find the python class corresponding to the module the savepoint is representing
    testobj = get_test_class_instance(test_name, split_name, args["serialized_grid"])
    if testobj is None:
        args["unimplemented"].append(savepoint_unique_name(sp))
        return
    if split_name[-1] == "In":
        process_input_savepoint(serializer, sp, testobj, test_name, args)
    else:
        process_output_savepoint(serializer, sp, testobj, test_name, args)
        args["testcount"] += 1


def process_grid_savepoint(serializer, sp_grid):
    import fv3._config

    serialized_grid = make_grid(sp_grid, serializer)
    # setting this so we don't have to pass it as an argument to every stencil
    fv3._config.set_grid(serialized_grid)
    return serialized_grid


# data savepoints are identified as <TestName>-In and <TestName>-out
def process_savepoint(serializer, sp, args):
    split_name = sp.name.split("-")
    if sp.name == "Grid-Info":
        args["sp_grid"] = sp
        args["serialized_grid"] = process_grid_savepoint(serializer, args["sp_grid"])
    elif split_name[-1] in ["In", "Out"]:
        process_test_savepoint(serializer, sp, split_name, args)
    else:
        # encountered savepoints not following our naming convention and thus we don't know how to test them
        args["unimplemented"].append(savepoint_unique_name(sp))


def test_serialized_savepoints(
    which_modules,
    skip_modules,
    print_failures,
    failure_stride,
    data_path,
    exec_backend,
    data_backend,
    backend,
):
    args = {
        "which_modules": which_modules,
        "skip_modules": skip_modules,
        "print_failures": print_failures,
        "failure_stride": failure_stride,
        "sp_grid": None,
        "serialized_grid": None,
        "testcount": 0,
    }
    if backend != "numpy":
        utils.exec_backend = backend
        utils.data_backend = backend
    else:
        utils.exec_backend = exec_backend
        utils.data_backend = data_backend
    for rank in range(6):
        serializer = ser.Serializer(
            ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
        )
        savepoints = serializer.savepoint_list()
        args.update(
            {
                "failure_stride": failure_stride,
                "isready": [],
                "unimplemented": [],
                "output_data": {},
                "rank": rank,
            }
        )
        print("Rank", args["rank"], "with", len(savepoints), "savepoints")
        # Iterate over all the savepoints in this mpi rank
        for sp in savepoints:
            try:
                process_savepoint(serializer, sp, args)
            except Exception as err:
                expect(
                    False,
                    msg="ERROR running stencil"
                    + sp.name
                    + "\n\t"
                    + str(err)
                    + "\n"
                    + traceback.format_exc(),
                )
                pass
        # lets make sure expectations are met for this rank, if not, no need to test all the other ranks
        missing_implementation_message = (
            bcolors.FAIL
            + "ERROR, there are savepoints encountered that do not have corresponding implemented python code!:\n\t"
            + "\n\t".join(args["unimplemented"])
            + bcolors.ENDC
        )
        # expect(len(unimplemented) == 0, msg=missing_implementation_message)
        assert_expectations()

        if len(args["unimplemented"]) > 0:
            print(missing_implementation_message)
    assert args["testcount"] > 0
