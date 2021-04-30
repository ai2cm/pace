# External Profiler

Add information for external profilers, by wrapping run into the `external_profiler.py` script.

Usage: `python external_profiler.py <PYTHON SCRIPT>.py <ARGS>`

## NVTX marking

Adds semantic marking using `nvtx` for `nsys` profiler (via `cupy`).

Configuration of the markings are inlined in code under the `functions_desc` list with:
* `fn`: function name to hook int.
* `file`: part of the source code file path, enough to form unique id with the `fn`.
* `name`: display name on the profiler. `str` or `Callable` with `(frame, event, args)->str` signature.


## Example usage on DAINT

`example.profiling.standalone.daint.slurm` is an example slurm job file to run the above. To be launched from
root director `fv3core` with a `sbatch` command.


# `nsys` kernel name lookup

Lookup kernel name using `nsys` correlation id and the `sqlite` database

Usage: `python nsys_id_kernel <NSYS_GENERATED.sqlite> <CORRELATION_ID> [--all]`

Correlation id can be found on the tooltip when overring above a kernel in `nsys` ui.
