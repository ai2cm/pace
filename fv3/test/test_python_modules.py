#!/usr/bin/env python3

import sys

sys.path.append("/serialbox2/install/python")  # noqa

import numpy as np
import serialbox as ser
import fv3._config
import fv3.utils.gt4py_utils
import pytest
import fv3util
import logging
import sys


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def collect_input_data(
    testobj, serializer, savepoint,
):
    input_data = {}
    for varname in (
        testobj.serialnames(testobj.in_vars["data_vars"])
        + testobj.in_vars["parameters"]
    ):
        input_data[varname] = read_serialized_data(serializer, savepoint, varname)
    return input_data


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
    input_data = collect_input_data(testobj, serializer, savepoint_in)
    # run python version of functionality
    output = testobj.compute(input_data)
    failing_names = []
    passing_names = []
    for varname in testobj.serialnames(testobj.out_vars):
        ref_data = read_serialized_data(serializer, savepoint_out, varname)
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


def get_communicator(comm, layout):
    partitioner = fv3util.CubedSpherePartitioner(fv3util.TilePartitioner(layout))
    communicator = fv3util.CubedSphereCommunicator(comm, partitioner)
    return communicator


@pytest.mark.parallel
def test_halo_update(data_path, subtests):
    n_ghost = fv3.utils.gt4py_utils.halo
    layout = fv3._config.namelist["layout"]
    total_ranks = 6 * layout[0] * layout[1]
    shared_buffer = {}
    states = []
    communicators = []
    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        savepoint = serializer.savepoint["HaloUpdate-In"]
        state = state_from_savepoint(
            serializer, savepoint, {"array": "air_temperature"}
        )
        states.append(state)
        comm = fv3util.testing.DummyComm(rank, total_ranks, buffer_dict=shared_buffer)
        communicator = get_communicator(comm, layout)
        communicator.start_halo_update(state["air_temperature"], n_ghost=n_ghost)
        communicators.append(communicator)
    for rank, (state, communicator) in enumerate(zip(states, communicators)):
        serializer = ser.Serializer(
            ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
        )
        savepoint = serializer.savepoint["HaloUpdate-Out"]
        array = serializer.read("array", savepoint)
        quantity = state["air_temperature"]
        communicator.finish_halo_update(quantity, n_ghost=n_ghost)
        with subtests.test(rank=rank):
            quantity.np.testing.assert_array_equal(quantity.data, array)
