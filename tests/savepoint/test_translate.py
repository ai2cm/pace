import contextlib
import hashlib
import logging
import os

import numpy as np
import pytest
import serialbox as ser
import xarray as xr

import fv3core._config
import fv3core.utils.gt4py_utils as gt_utils
import fv3gfs.util as fv3util
from fv3core.utils.mpi import MPI


# this only matters for manually-added print statements
np.set_printoptions(threshold=4096)

OUTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
GPU_MAX_ERR = 1e-10
GPU_NEAR_ZERO = 1e-15


def compare_arr(computed_data, ref_data):
    denom = np.abs(ref_data) + np.abs(computed_data)
    compare = 2.0 * np.abs(computed_data - ref_data) / denom
    compare[denom == 0] = 0.0
    return compare


def success_array(computed_data, ref_data, eps, ignore_near_zero_errors, near_zero):
    success = np.logical_or(
        np.logical_and(np.isnan(computed_data), np.isnan(ref_data)),
        compare_arr(computed_data, ref_data) < eps,
    )
    if ignore_near_zero_errors:
        success = np.logical_or(
            success,
            np.logical_and(
                np.abs(computed_data) < near_zero, np.abs(ref_data) < near_zero
            ),
        )
    return success


def success(computed_data, ref_data, eps, ignore_near_zero_errors, near_zero=0.0):
    return np.all(
        success_array(computed_data, ref_data, eps, ignore_near_zero_errors, near_zero)
    )


def platform():
    in_docker = os.environ.get("IN_DOCKER", False)
    return "docker" if in_docker else "metal"


def sample_wherefail(
    computed_data,
    ref_data,
    eps,
    print_failures,
    failure_stride,
    test_name,
    ignore_near_zero_errors,
    near_zero,
    xy_indices=False,
):
    found_indices = np.where(
        np.logical_not(
            success_array(
                computed_data, ref_data, eps, ignore_near_zero_errors, near_zero
            )
        )
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

    if xy_indices:
        if len(computed_data.shape) == 3:
            axis = 2
            any = np.any
        elif len(computed_data.shape) == 4:
            axis = (2, 3)
            any = np.any
        else:
            axis = None

            def any(array, axis):
                return array

        found_xy_indices = np.where(
            any(
                np.logical_not(
                    success_array(
                        computed_data, ref_data, eps, ignore_near_zero_errors, near_zero
                    )
                ),
                axis=axis,
            )
        )

        return_strings.append(
            "failed horizontal indices:" + str(list(zip(*found_xy_indices)))
        )

    return "\n".join(return_strings)


def process_override(threshold_overrides, testobj, test_name, backend):
    override = threshold_overrides.get(test_name, None)
    if override is not None:
        for spec in override:
            if "platform" not in spec:
                spec["platform"] = platform()
            if "backend" not in spec:
                spec["backend"] = backend
        matches = [
            spec
            for spec in override
            if spec["backend"] == backend and spec["platform"] == platform()
        ]
        if len(matches) == 1:
            match = matches[0]
            if "max_error" in match:
                testobj.max_error = float(match["max_error"])
            if "near_zero" in match:
                testobj.near_zero = float(match["near_zero"])
            if "ignore_near_zero_errors" in match:
                testobj.ignore_near_zero_errors = {
                    field: True for field in match["ignore_near_zero_errors"]
                }
        elif len(matches) > 1:
            raise Exception(
                "misconfigured threshold overrides file, more than 1 specification for "
                + test_name
                + " with backend="
                + backend
                + ", platform="
                + platform()
            )


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
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
    threshold_overrides,
    xy_indices=True,
):
    caplog.set_level(logging.DEBUG, logger="fv3core")
    if testobj is None:
        pytest.xfail(f"no translate object available for savepoint {test_name}")
    # Reduce error threshold for GPU
    if backend.endswith("cuda"):
        testobj.max_error = max(testobj.max_error, GPU_MAX_ERR)
        testobj.near_zero = max(testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(threshold_overrides, testobj, test_name, backend)
    fv3core._config.set_grid(grid)
    input_data = testobj.collect_input_data(serializer, savepoint_in)
    # run python version of functionality
    output = testobj.compute(input_data)
    failing_names = []
    passing_names = []
    for varname in testobj.serialnames(testobj.out_vars):
        ignore_near_zero = testobj.ignore_near_zero_errors.get(varname, False)
        ref_data = serializer.read(varname, savepoint_out)
        if hasattr(testobj, "subset_output"):
            ref_data = testobj.subset_output(varname, ref_data)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            assert success(
                output[varname],
                ref_data,
                testobj.max_error,
                ignore_near_zero,
                testobj.near_zero,
            ), sample_wherefail(
                output[varname],
                ref_data,
                testobj.max_error,
                print_failures,
                failure_stride,
                test_name,
                ignore_near_zero_errors=ignore_near_zero,
                near_zero=testobj.near_zero,
                xy_indices=xy_indices,
            )
            passing_names.append(failing_names.pop())
    assert failing_names == [], f"only the following variables passed: {passing_names}"
    assert len(passing_names) > 0, "No tests passed"


def get_serializer(data_path, rank):
    return ser.Serializer(
        ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
    )


def state_from_savepoint(serializer, savepoint, name_to_std_name):
    properties = fv3util.fortran_info.properties_by_std_name
    origin = gt_utils.origin
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


@pytest.mark.mock_parallel
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
def test_mock_parallel_savepoint(
    testobj,
    test_name,
    grid,
    mock_communicator_list,
    serializer_list,
    savepoint_in_list,
    savepoint_out_list,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    threshold_overrides,
    xy_indices=False,
):
    caplog.set_level(logging.DEBUG, logger="fv3core")
    caplog.set_level(logging.DEBUG, logger="fv3util")
    if testobj is None:
        pytest.xfail(f"no translate object available for savepoint {test_name}")
    # Reduce error threshold for GPU
    if backend.endswith("cuda"):
        testobj.max_error = max(testobj.max_error, GPU_MAX_ERR)
        testobj.near_zero = max(testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(threshold_overrides, testobj, test_name, backend)
    fv3core._config.set_grid(grid)
    inputs_list = []
    for savepoint_in, serializer in zip(savepoint_in_list, serializer_list):
        inputs_list.append(testobj.collect_input_data(serializer, savepoint_in))
    output_list = testobj.compute_sequential(inputs_list, mock_communicator_list)
    failing_names = []
    ref_data = {}
    for varname in testobj.outputs.keys():
        ref_data[varname] = []
        ignore_near_zero = testobj.ignore_near_zero_errors.get(varname, False)
        with _subtest(failing_names, subtests, varname=varname):
            failing_ranks = []
            for rank, (savepoint_out, serializer, output) in enumerate(
                zip(savepoint_out_list, serializer_list, output_list)
            ):
                with _subtest(failing_ranks, subtests, varname=varname, rank=rank):
                    ref_data[varname].append(serializer.read(varname, savepoint_out))
                    assert success(
                        gt_utils.asarray(output[varname]),
                        ref_data[varname][-1],
                        testobj.max_error,
                        ignore_near_zero,
                        testobj.near_zero,
                    ), sample_wherefail(
                        output[varname],
                        ref_data[varname][-1],
                        testobj.max_error,
                        print_failures,
                        failure_stride,
                        test_name,
                        ignore_near_zero,
                        testobj.near_zero,
                        xy_indices,
                    )
            assert failing_ranks == []
    failing_names = [item["varname"] for item in failing_names]
    if len(failing_names) > 0:
        out_filename = os.path.join(OUTDIR, f"{test_name}.nc")
        try:
            save_netcdf(
                testobj, inputs_list, output_list, ref_data, failing_names, out_filename
            )
        except Exception as error:
            print(error)
    assert failing_names == [], f"names tested: {list(testobj.outputs.keys())}"


def hash_result_data(result, data_keys):
    hashes = {}
    for k in data_keys:
        hashes[k] = hashlib.sha1(
            np.ascontiguousarray(gt_utils.asarray(result[k]))
        ).hexdigest()
    return hashes


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_parallel_savepoint(
    data_regression,
    data_path,
    testobj,
    test_name,
    test_case,
    grid,
    serializer,
    savepoint_in,
    savepoint_out,
    communicator,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    python_regression,
    threshold_overrides,
    xy_indices=True,
):
    caplog.set_level(logging.DEBUG, logger="fv3core")
    if python_regression and not testobj.python_regression:
        pytest.xfail(f"python_regression not set for test {test_name}")
    if testobj is None:
        pytest.xfail(f"no translate object available for savepoint {test_name}")
    # Increase minimum error threshold for GPU
    if backend.endswith("cuda"):
        testobj.max_error = max(testobj.max_error, GPU_MAX_ERR)
        testobj.near_zero = max(testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(threshold_overrides, testobj, test_name, backend)
    fv3core._config.set_grid(grid[0])
    input_data = testobj.collect_input_data(serializer, savepoint_in)
    # run python version of functionality
    output = testobj.compute_parallel(input_data, communicator)
    out_vars = set(testobj.outputs.keys())
    out_vars.update(list(testobj._base.out_vars.keys()))
    if python_regression and testobj.python_regression:
        filename = f"python_regressions/{test_case}_{backend}_{platform()}.yml"
        filename = filename.replace("=", "_")
        data_regression.check(
            hash_result_data(output, out_vars),
            fullpath=os.path.join(data_path, filename),
        )
        return
    failing_names = []
    passing_names = []
    ref_data = {}
    for varname in out_vars:
        ref_data[varname] = []
        new_ref_data = serializer.read(varname, savepoint_out)
        if hasattr(testobj, "subset_output"):
            new_ref_data = testobj.subset_output(varname, new_ref_data)
        ref_data[varname].append(new_ref_data)
        ignore_near_zero = testobj.ignore_near_zero_errors.get(varname, False)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            assert success(
                output[varname],
                ref_data[varname][0],
                testobj.max_error,
                ignore_near_zero,
                testobj.near_zero,
            ), sample_wherefail(
                output[varname],
                ref_data[varname][0],
                testobj.max_error,
                print_failures,
                failure_stride,
                test_name,
                ignore_near_zero,
                testobj.near_zero,
                xy_indices,
            )
            passing_names.append(failing_names.pop())
    if len(failing_names) > 0:
        out_filename = os.path.join(OUTDIR, f"{test_name}-{grid[0].rank}.nc")
        try:
            save_netcdf(
                testobj, [input_data], [output], ref_data, failing_names, out_filename
            )
        except Exception as error:
            print(error)
    assert failing_names == [], f"only the following variables passed: {passing_names}"
    assert len(passing_names) > 0, "No tests passed"


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
    for i, varname in enumerate(failing_names):
        dims = [dim_name + f"_{i}" for dim_name in testobj.outputs[varname]["dims"]]
        attrs = {"units": testobj.outputs[varname]["units"]}
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
