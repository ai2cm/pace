import copy
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pytest

import pace.dsl
import pace.dsl.gt4py_utils as gt_utils
import pace.util
from pace.dsl.dace.dace_config import DaceConfig
from pace.dsl.stencil import CompilationConfig
from pace.stencils.testing import SavepointCase, dataset_to_dict
from pace.util.mpi import MPI
from pace.util.testing import compare_scalar, perturb, success, success_array


# this only matters for manually-added print statements
np.set_printoptions(threshold=4096)

OUTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
GPU_MAX_ERR = 1e-10
GPU_NEAR_ZERO = 1e-15


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

    # List all errors
    return_strings = []
    bad_indices_count = len(found_indices[0])
    # Determine worst result
    worst_metric_err = 0.0
    for b in range(bad_indices_count):
        full_index = [f[b] for f in found_indices]
        metric_err = compare_scalar(computed_failures[b], reference_failures[b])
        abs_err = abs(computed_failures[b] - reference_failures[b])
        if print_failures and b % failure_stride == 0:
            return_strings.append(
                f"index: {full_index}, computed {computed_failures[b]}, "
                f"reference {reference_failures[b]}, "
                f"absolute diff {abs_err:.3e}, "
                f"metric diff: {metric_err:.3e}"
            )
        if np.isnan(metric_err) or (metric_err > worst_metric_err):
            worst_metric_err = metric_err
            worst_full_idx = full_index
            worst_abs_err = abs_err
            computed_worst = computed_failures[b]
            reference_worst = reference_failures[b]
    # Summary and worst result
    fullcount = len(ref_data.flatten())
    return_strings.append(
        f"Failed count: {bad_indices_count}/{fullcount} "
        f"({round(100.0 * (bad_indices_count / fullcount), 2)}%),\n"
        f"Worst failed index {worst_full_idx}\n"
        f"\tcomputed:{computed_worst}\n"
        f"\treference: {reference_worst}\n"
        f"\tabsolute diff: {worst_abs_err:.3e}\n"
        f"\tmetric diff: {worst_metric_err:.3e}\n"
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
                parsed_ignore_zero = match["ignore_near_zero_errors"]
                if isinstance(parsed_ignore_zero, list):
                    testobj.ignore_near_zero_errors.update(
                        {field: True for field in match["ignore_near_zero_errors"]}
                    )
                elif isinstance(parsed_ignore_zero, dict):
                    for key in parsed_ignore_zero.keys():
                        testobj.ignore_near_zero_errors[key] = {}
                        testobj.ignore_near_zero_errors[key]["near_zero"] = float(
                            parsed_ignore_zero[key]
                        )
                    if "all_other_near_zero" in match:
                        for key in testobj.out_vars.keys():
                            if key not in testobj.ignore_near_zero_errors:
                                testobj.ignore_near_zero_errors[key] = {}
                                testobj.ignore_near_zero_errors[key][
                                    "near_zero"
                                ] = float(match["all_other_near_zero"])

                else:
                    raise TypeError(
                        "ignore_near_zero_errors is either a list or a dict"
                    )
        elif len(matches) > 1:
            raise Exception(
                "misconfigured threshold overrides file, more than 1 specification for "
                + test_name
                + " with backend="
                + backend
                + ", platform="
                + platform()
            )


N_THRESHOLD_SAMPLES = 10


def get_thresholds(testobj, input_data):
    return _get_thresholds(testobj.compute, input_data)


def get_thresholds_parallel(testobj, input_data, communicator):
    def compute(input):
        return testobj.compute_parallel(input, communicator)

    return _get_thresholds(compute, input_data)


def _get_thresholds(compute_function, input_data):
    output_list = []
    for _ in range(N_THRESHOLD_SAMPLES):
        input = copy.deepcopy(input_data)
        perturb(input)
        output_list.append(compute_function(input))

    output_varnames = output_list[0].keys()
    for varname in output_varnames:
        if output_list[0][varname].dtype in (
            np.float64,
            np.int64,
            np.float32,
            np.int32,
        ):
            samples = [out[varname] for out in output_list]
            pointwise_max_abs_errors = np.max(samples, axis=0) - np.min(samples, axis=0)
            max_rel_diff = np.nanmax(
                pointwise_max_abs_errors / np.min(np.abs(samples), axis=0)
            )
            max_abs_diff = np.nanmax(pointwise_max_abs_errors)
            print(
                f"{varname}: max rel diff {max_rel_diff}, max abs diff {max_abs_diff}"
            )


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
def test_sequential_savepoint(
    case: SavepointCase,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    threshold_overrides,
    xy_indices=True,
):
    caplog.set_level(logging.DEBUG, logger="fv3core")
    if case.testobj is None:
        pytest.xfail(
            f"no translate object available for savepoint {case.savepoint_name}"
        )
    stencil_config = pace.dsl.StencilConfig(
        compilation_config=CompilationConfig(backend=backend),
        dace_config=DaceConfig(
            communicator=None,
            backend=backend,
        ),
    )
    # Reduce error threshold for GPU
    if stencil_config.is_gpu_backend:
        case.testobj.max_error = max(case.testobj.max_error, GPU_MAX_ERR)
        case.testobj.near_zero = max(case.testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(
            threshold_overrides, case.testobj, case.savepoint_name, backend
        )
    input_data = dataset_to_dict(case.ds_in)
    input_names = (
        case.testobj.serialnames(case.testobj.in_vars["data_vars"])
        + case.testobj.in_vars["parameters"]
    )
    input_data = {name: input_data[name] for name in input_names}
    original_input_data = copy.deepcopy(input_data)
    # run python version of functionality
    output = case.testobj.compute(input_data)
    failing_names: List[str] = []
    passing_names: List[str] = []
    all_ref_data = dataset_to_dict(case.ds_out)
    ref_data_out = {}
    for varname in case.testobj.serialnames(case.testobj.out_vars):
        ignore_near_zero = case.testobj.ignore_near_zero_errors.get(varname, False)
        ref_data = all_ref_data[varname]
        if hasattr(case.testobj, "subset_output"):
            ref_data = case.testobj.subset_output(varname, ref_data)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            output_data = gt_utils.asarray(output[varname])
            assert success(
                output_data,
                ref_data,
                case.testobj.max_error,
                ignore_near_zero,
                case.testobj.near_zero,
            ), sample_wherefail(
                output_data,
                ref_data,
                case.testobj.max_error,
                print_failures,
                failure_stride,
                case.savepoint_name,
                ignore_near_zero_errors=ignore_near_zero,
                near_zero=case.testobj.near_zero,
                xy_indices=xy_indices,
            )
            passing_names.append(failing_names.pop())
        ref_data_out[varname] = [ref_data]
    if len(failing_names) > 0:
        get_thresholds(case.testobj, input_data=original_input_data)
        out_filename = os.path.join(OUTDIR, f"{case.savepoint_name}.nc")
        save_netcdf(
            case.testobj,
            [input_data],
            [output],
            ref_data_out,
            failing_names,
            out_filename,
        )
    assert failing_names == [], f"only the following variables passed: {passing_names}"
    assert len(passing_names) > 0, "No tests passed"


def state_from_savepoint(serializer, savepoint, name_to_std_name):
    properties = pace.util.fortran_info.properties_by_std_name
    origin = gt_utils.origin
    state = {}
    for name, std_name in name_to_std_name.items():
        array = serializer.read(name, savepoint)
        extent = tuple(np.asarray(array.shape) - 2 * np.asarray(origin))
        state["air_temperature"] = pace.util.Quantity(
            array,
            dims=reversed(properties["air_temperature"]["dims"]),
            units=properties["air_temperature"]["units"],
            origin=origin,
            extent=extent,
        )
    return state


def get_communicator(comm, layout):
    partitioner = pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout))
    communicator = pace.util.CubedSphereCommunicator(comm, partitioner)
    return communicator


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_parallel_savepoint(
    case: SavepointCase,
    backend,
    print_failures,
    failure_stride,
    subtests,
    caplog,
    threshold_overrides,
    compute_grid,
    xy_indices=True,
):
    layout = (
        int((MPI.COMM_WORLD.Get_size() // 6) ** 0.5),
        int((MPI.COMM_WORLD.Get_size() // 6) ** 0.5),
    )
    communicator = get_communicator(MPI.COMM_WORLD, layout)
    caplog.set_level(logging.DEBUG, logger="fv3core")
    if case.testobj is None:
        pytest.xfail(
            f"no translate object available for savepoint {case.savepoint_name}"
        )
    stencil_config = pace.dsl.StencilConfig(
        compilation_config=CompilationConfig(backend=backend),
        dace_config=DaceConfig(
            communicator=communicator,
            backend=backend,
        ),
    )
    # Increase minimum error threshold for GPU
    if stencil_config.is_gpu_backend:
        case.testobj.max_error = max(case.testobj.max_error, GPU_MAX_ERR)
        case.testobj.near_zero = max(case.testobj.near_zero, GPU_NEAR_ZERO)
    if threshold_overrides is not None:
        process_override(
            threshold_overrides, case.testobj, case.savepoint_name, backend
        )
    if compute_grid and not case.testobj.compute_grid_option:
        pytest.xfail(f"compute_grid option not used for test {case.savepoint_name}")
    input_data = dataset_to_dict(case.ds_in)
    # run python version of functionality
    output = case.testobj.compute_parallel(input_data, communicator)
    out_vars = set(case.testobj.outputs.keys())
    out_vars.update(list(case.testobj._base.out_vars.keys()))
    failing_names = []
    passing_names = []
    ref_data: Dict[str, Any] = {}
    all_ref_data = dataset_to_dict(case.ds_out)
    for varname in out_vars:
        ref_data[varname] = []
        new_ref_data = all_ref_data[varname]
        if hasattr(case.testobj, "subset_output"):
            new_ref_data = case.testobj.subset_output(varname, new_ref_data)
        ref_data[varname].append(new_ref_data)
        ignore_near_zero = case.testobj.ignore_near_zero_errors.get(varname, False)
        with subtests.test(varname=varname):
            failing_names.append(varname)
            output_data = gt_utils.asarray(output[varname])
            assert success(
                output_data,
                ref_data[varname][0],
                case.testobj.max_error,
                ignore_near_zero,
                case.testobj.near_zero,
            ), sample_wherefail(
                output_data,
                ref_data[varname][0],
                case.testobj.max_error,
                print_failures,
                failure_stride,
                case.savepoint_name,
                ignore_near_zero,
                case.testobj.near_zero,
                xy_indices,
            )
            passing_names.append(failing_names.pop())
    if len(failing_names) > 0:
        out_filename = os.path.join(
            OUTDIR, f"{case.savepoint_name}-{case.grid.rank}.nc"
        )
        try:
            save_netcdf(
                case.testobj,
                [input_data],
                [output],
                ref_data,
                failing_names,
                out_filename,
            )
        except Exception as error:
            print(f"TestParallel SaveNetCDF Error: {error}")
    assert failing_names == [], f"only the following variables passed: {passing_names}"
    assert len(passing_names) > 0, "No tests passed"


def save_netcdf(
    testobj,
    # first list over rank, second list over savepoint
    inputs_list: List[Dict[str, List[np.ndarray]]],
    output_list: List[Dict[str, List[np.ndarray]]],
    ref_data: Dict[str, List[np.ndarray]],
    failing_names,
    out_filename,
):
    import xarray as xr

    data_vars = {}
    for i, varname in enumerate(failing_names):
        if hasattr(testobj, "outputs"):
            dims = [dim_name + f"_{i}" for dim_name in testobj.outputs[varname]["dims"]]
            attrs = {"units": testobj.outputs[varname]["units"]}
        else:
            dims = [
                f"dim_{varname}_{j}" for j in range(len(ref_data[varname][0].shape))
            ]
            attrs = {"units": "unknown"}
        try:
            data_vars[f"{varname}_in"] = xr.DataArray(
                np.stack([in_data[varname] for in_data in inputs_list]),
                dims=("rank",) + tuple([f"{d}_in" for d in dims]),
                attrs=attrs,
            )
        except KeyError as error:
            print(f"No input data found for {error}")
        data_vars[f"{varname}_ref"] = xr.DataArray(
            np.stack(ref_data[varname]),
            dims=("rank",) + tuple([f"{d}_out" for d in dims]),
            attrs=attrs,
        )
        data_vars[f"{varname}_out"] = xr.DataArray(
            np.stack([output[varname] for output in output_list]),
            dims=("rank",) + tuple([f"{d}_out" for d in dims]),
            attrs=attrs,
        )
        data_vars[f"{varname}_error"] = (
            data_vars[f"{varname}_ref"] - data_vars[f"{varname}_out"]
        )
        data_vars[f"{varname}_error"].attrs = attrs
    print(f"File saved to {out_filename}")
    xr.Dataset(data_vars=data_vars).to_netcdf(out_filename)
