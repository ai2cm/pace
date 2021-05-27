# External Profiler

External tooling for profiling code.
It works by wrapping runs into the `external_profiler.py` script.

Usage: `python external_profiler.py <ARGS> <TARGET PYTHON SCRIPT>.py <TARGET ARGS>`

## NVTX marking (`--nvtx`)

Adds semantic marking using `nvtx` for `nsys` profiler (via `cupy`).

Configuration of the markings are inlined in code under the `functions_desc` list with:
* `fn`: function name to hook int.
* `file`: part of the source code file path, enough to form unique id with the `fn`.
* `name`: display name on the profiler. `str` or `Callable` with `(frame, event, args)->str` signature.

Example command line: `python external_profiler.py --nvtx <TARGET PYTHON SCRIPT>.py <TARGET ARGS>`

### Example usage on DAINT

`example.profiling.standalone.daint.slurm` is an example slurm job file to run the above. To be launched from
root director `fv3core` with a `sbatch` command.

## Stencil reproducer (`--stencil=STENCIL_NAME`)

Capture all data, libs & source code to reproduce a stencil run on its own. It requires the targeted stencil to be already
compiled.

Once ran the reproducer package all required files in a `repro_<STENCIL_NAME>_<TIMESTAMP>` folder.

The reproducer will also write a `repro.py` script that runs a single computation of the stencil.

Example command line: `python external_profiler.py --stencil=MY_STENCIL <TARGET PYTHON SCRIPT>.py <TARGET ARGS>`

### Using `ncu`

This work has started in order to allow easy run on `ncu`. And `example_run_ncu.sh` is provided as an example to show
how to run the captured stencil with `ncu`.


## Other tools:

# `nsys` kernel name lookup

Lookup kernel name using `nsys` correlation id and the `sqlite` database

Usage: `python nsys_id_kernel <NSYS_GENERATED.sqlite> <CORRELATION_ID> [--all]`

Correlation id can be found on the tooltip when overring above a kernel in `nsys` ui.
