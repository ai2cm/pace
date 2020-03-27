#!/usr/bin/env python3

import sys
import contextlib
import numpy as np
import fv3._config
import fv3.utils.gt4py_utils
import pytest
import fv3util
import logging
import os
import xarray as xr

sys.path.append("/serialbox2/install/python")  # noqa
import serialbox as ser

# this only matters for manually-added print statements
np.set_printoptions(threshold=4096)

OUTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")


def compare_arr(computed_data, ref_data):
    denom = np.abs(ref_data) + np.abs(computed_data)
    compare = 2.0 * np.abs(computed_data - ref_data) / denom
    compare[denom == 0] = 0.0
    return compare


def success_array(computed_data, ref_data, eps):
    return np.logical_or(
        compare_arr(computed_data, ref_data) < eps,
        np.logical_and(np.isnan(computed_data), np.isnan(ref_data)),
    )


def success(computed_data, ref_data, eps):
    return np.all(success_array(computed_data, ref_data, eps))


def sample_wherefail(
    computed_data, ref_data, eps, print_failures, failure_stride, test_name
):
    found_indices = np.where(
        np.logical_not(success_array(computed_data, ref_data, eps))
    )
    computed_failures = computed_data[found_indices]
    reference_failures = ref_data[found_indices]
    return_strings = []
    bad_indices_count = len(found_indices[0])
    if print_failures:
        for b in range(0, bad_indices_count, failure_stride):
            full_index = [f[b] for f in found_indices]
            return_strings.append(
                f"index: {full_index}, computed {computed_failures[b]}, "
                f"reference {reference_failures[b]}, "
                f"diff {abs(computed_failures[b] - reference_failures[b])}"
            )
    sample = [f[0] for f in found_indices]
    fullcount = len(ref_data.flatten())
    return_strings.append(
        f"Failed count: {bad_indices_count}/{fullcount} "
        f"({round(100.0 * (bad_indices_count / fullcount), 2)}%),\n"
        f"first failed index {sample}, computed:{computed_failures[0]}, "
        f"reference: {reference_failures[0]}, "
        f"diff: {abs(computed_failures[0] - reference_failures[0])}\n"
    )
    return "\n".join(return_strings)


@pytest.mark.sequential
def test_sequential_savepoint(
    testobj,
    test_name,
    grid,
    serializer,
    savepoint_in,
    savepoint_out,
    rank,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
):
    caplog.set_level(logging.DEBUG, logger="fv3ser")
    if testobj is None:
        pytest.xfail(f"no translate object available for savepoint {test_name}")
    fv3._config.set_grid(grid)
    input_data = testobj.collect_input_data(serializer, savepoint_in)
    # run python version of functionality
    output = testobj.compute(input_data)
    failing_names = []
    passing_names = []
    for varname in testobj.serialnames(testobj.out_vars):
        ref_data = serializer.read(varname, savepoint_out)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            assert success(
                output[varname], ref_data, testobj.max_error
            ), sample_wherefail(
                output[varname],
                ref_data,
                testobj.max_error,
                print_failures,
                failure_stride,
                test_name,
            )
            passing_names.append(failing_names.pop())
    assert failing_names == [], f"only the following variables passed: {passing_names}"


def get_serializer(data_path, rank):
    return ser.Serializer(
        ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
    )


def state_from_savepoint(serializer, savepoint, name_to_std_name):
    properties = fv3util.fortran_info.properties_by_std_name
    origin = fv3.utils.gt4py_utils.origin
    state = {}
    for name, std_name in name_to_std_name.items():
        array = serializer.read(name, savepoint)
        extent = tuple(np.asarray(array.shape) - 2 * np.asarray(origin))
        state["air_temperature"] = fv3util.Quantity(
            array,
            dims=reversed(properties["air_temperature"]["dims"]),
            units=properties["air_temperature"]["units"],
            origin=origin,
            extent=extent,
        )
    return state


@pytest.mark.parallel
def test_parallel_savepoint_sequentially(
    testobj,
    test_name,
    grid,
    communicator_list,
    serializer_list,
    savepoint_in_list,
    savepoint_out_list,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
):
    caplog.set_level(logging.DEBUG, logger="fv3ser")
    caplog.set_level(logging.DEBUG, logger="fv3util")
    if testobj is None:
        pytest.xfail(f"no translate object available for savepoint {test_name}")
    fv3._config.set_grid(grid)
    inputs_list = []
    for savepoint_in, serializer in zip(savepoint_in_list, serializer_list):
        inputs_list.append(testobj.collect_input_data(serializer, savepoint_in))
    output_list = testobj.compute_sequential(inputs_list, communicator_list)
    failing_names = []
    ref_data = {}
    for varname in testobj.outputs.keys():
        ref_data[varname] = []
        with _subtest(failing_names, subtests, varname=varname):
            with subtests.test(varname=varname):
                failing_ranks = []
                for rank, (savepoint_out, serializer, output) in enumerate(
                    zip(savepoint_out_list, serializer_list, output_list)
                ):
                    with _subtest(failing_ranks, subtests, varname=varname, rank=rank):
                        ref_data[varname].append(
                            serializer.read(varname, savepoint_out)
                        )
                        assert success(
                            output[varname], ref_data[varname][-1], testobj.max_error
                        ), sample_wherefail(
                            output[varname],
                            ref_data[varname][-1],
                            testobj.max_error,
                            print_failures,
                            failure_stride,
                            test_name,
                        )
                assert failing_ranks == []
    if len(failing_names) > 0:
        out_filename = os.path.join(OUTDIR, f"{test_name}.nc")
        save_netcdf(
            testobj, inputs_list, output_list, ref_data, failing_names, out_filename
        )
    assert failing_names == [], f"names tested: {list(testobj.outputs.keys())}"


@contextlib.contextmanager
def _subtest(failure_list, subtests, **kwargs):
    failure_list.append(kwargs)
    with subtests.test(**kwargs):
        yield
    failure_list.pop()  # will remove kwargs if the test passes


def save_netcdf(
    testobj, inputs_list, output_list, ref_data, failing_names, out_filename
):
    data_vars = {}
    for varname in failing_names:
        dims = testobj.outputs[varname]["dims"]
        attrs = {
            "units": testobj.outputs[varname]["units"],
            "n_halo": testobj.outputs[varname]["n_halo"],
        }
        data_vars[f"{varname}_in"] = xr.DataArray(
            np.stack([in_data[varname] for in_data in inputs_list]),
            dims=("rank",) + tuple(dims),
            attrs=attrs,
        )
        data_vars[f"{varname}_ref"] = xr.DataArray(
            np.stack(ref_data[varname]), dims=("rank",) + tuple(dims), attrs=attrs
        )
        data_vars[f"{varname}_out"] = xr.DataArray(
            np.stack([output[varname] for output in output_list]),
            dims=("rank",) + tuple(dims),
            attrs=attrs,
        )
        data_vars[f"{varname}_error"] = (
            data_vars[f"{varname}_ref"] - data_vars[f"{varname}_out"]
        )
        data_vars[f"{varname}_error"].attrs = attrs
    xr.Dataset(data_vars=data_vars).to_netcdf(out_filename)
