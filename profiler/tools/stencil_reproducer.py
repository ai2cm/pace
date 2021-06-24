import fnmatch
import pickle
from datetime import datetime
from glob import glob
from os import getcwd, getenv, listdir, mkdir, path, walk
from shutil import copy, copytree
from typing import Dict, Tuple

import numpy as np
from gt4py import storage

from fv3core.utils.mpi import MPI


STENCIL_CANDIDATE_FOR_EXTRACT: Dict[str, Tuple[str, str]] = {}


def field_serialization(frame, event, args):
    """Serialize all fields from a stencil"""
    if event == "call" or event == "return":
        for stencil_key, stencil_info in STENCIL_CANDIDATE_FOR_EXTRACT.items():
            # search for the stencil independent of the cache directory
            # under {backend}/fv3core/decorators/{stencil}/
            end_of_stencil_path = "/".join(stencil_info[0].split("/")[-5:])
            if (
                frame.f_code.co_name == "run"
                and end_of_stencil_path in frame.f_code.co_filename
            ):
                print(f"[PROFILER] Pickling args of {stencil_key} @ event {event}")
                if event == "call":
                    prefix = "pre_run_"
                else:
                    prefix = "post_run_"

                scalars = {}
                for arg_key, arg_value in frame.f_locals.items():
                    if arg_key == "self":
                        continue
                    if isinstance(arg_value, storage.Storage):
                        arg_value.device_to_host()
                        pickle_file = f"{stencil_info[1]}/data/{prefix}_{arg_key}.npz"
                        if path.isfile(pickle_file):
                            # TODO: sensible handling for args.call_number > 0
                            print(f"already wrote to {pickle_file}, skipping...")
                            return -1
                        np.savez_compressed(
                            pickle_file,
                            arg_value.data,
                        )
                    else:
                        scalars[arg_key] = arg_value
                scalar_file = f"{stencil_info[1]}/data/{prefix}_scalars.pickled"
                with open(scalar_file, "wb") as handle:
                    pickle.dump(scalars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def collect_stencil_candidate(stencil_name, call_number):
    """Collect all stencils that correspond to `stencil_name`.

    Multiple stencils can be collected if compile time varaiables lead to multiple
    stencil code path.
    """
    # Collect all compiled version of the stencil
    expected_py_wrapper_partialname = f"m_{stencil_name}__"
    gt_cache_root = getenv("GT_CACHE_ROOT")
    gt_cache_root = gt_cache_root if gt_cache_root is not None else getcwd()
    print(f"[PROFILER] Searching for {stencil_name} in {gt_cache_root}...")
    for fname in listdir(gt_cache_root):
        fullpath = path.join(gt_cache_root, fname)
        if fname.startswith(".gt_cache") and path.isdir(fullpath):
            for root, _, filenames in walk(fullpath):
                for call_count, py_wrapper_file in enumerate(
                    fnmatch.filter(filenames, f"{expected_py_wrapper_partialname}*.py")
                ):
                    if (call_number <= 0) or (call_count + 1 == call_number):
                        print(f"...found candidate {path.join(root, py_wrapper_file)}")
                        stencil_key = path.splitext(py_wrapper_file)[0]
                        stencil_file_wrapper = path.join(root, py_wrapper_file)
                        STENCIL_CANDIDATE_FOR_EXTRACT[stencil_key] = (
                            stencil_file_wrapper,
                            None,
                        )

    # Raise an exception is the collection came back with no results
    if len(STENCIL_CANDIDATE_FOR_EXTRACT.items()) == 0:
        raise RuntimeError(f"[Profiler] not stencil collected for {stencil_name}")

    # Create the result dir
    rank = MPI.COMM_WORLD.Get_rank()
    repro_dir = (
        f"{getcwd()}/repro_{stencil_name}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')}"
        f"{rank}"
    )
    mkdir(repro_dir)
    # Copy required file for repro & prepare for args pickling
    for stencil_key, stencil_info in STENCIL_CANDIDATE_FOR_EXTRACT.items():
        # One folder per stencil candidate
        stencil_dir = f"{repro_dir}/{stencil_key}"
        mkdir(stencil_dir)
        # Save info
        stencil_info = (stencil_info[0], stencil_dir)
        STENCIL_CANDIDATE_FOR_EXTRACT[stencil_key] = stencil_info
        # Create data directories
        mkdir(f"{stencil_dir}/data")
        # Copy original code
        origin_code_copy_dir = f"{stencil_dir}/original_code"
        mkdir(origin_code_copy_dir)
        widlcard = f"{path.dirname(stencil_info[0])}/{stencil_key[:-2]}*"
        for orignal_file in glob(widlcard):
            if path.isfile(orignal_file):
                copy(orignal_file, origin_code_copy_dir)
            if path.isdir(orignal_file):
                copytree(orignal_file, origin_code_copy_dir, dirs_exist_ok=True)
        # Write reproducer script
        with open(f"{stencil_dir}/repro.py", "w") as handle:
            handle.write(
                f"""from original_code import {stencil_key}
from os import path
import pickle
from gt4py import storage
import cupy as cp
import numpy as np

if __name__ == "__main__":
    # Load compiled object
    root_dir = path.dirname(path.realpath(__file__))
    compute_object = (
        {stencil_key}.{stencil_key[2:].replace('__', '____')}()
    )
    # Select a module depending on backend to load the serialized data
    loading_module = np
    if compute_object._gt_backend_ == "gtcuda":
        loading_module = cp
    # Setup the fields
    arguments = {{}}
    for field_name, _field_info in compute_object._gt_field_info_.items():
        field_file = f"{{root_dir}}/data/pre_run__{{field_name}}.npz"
        with cp.load(field_file) as npz_handle:
            arguments[field_name] = storage.from_array(
                npz_handle["arr_0"], compute_object._gt_backend_, (0, 0, 0)
            )
    # Un-pickle the scalars and finalize the argument list
    with open(root_dir + "/data/pre_run__scalars.pickled", "rb") as handle:
        arguments.update(pickle.load(handle))
    compute_object.run(**arguments)
"""
            )
