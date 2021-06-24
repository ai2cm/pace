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

If multiple versions of the stencil are called the reproducer will extract all of them. To select the N-th single stencil version add `--call_number=N`.

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

# `nsys` datbase mining

`nsys` can generate a `sqlite` database from its internal `qdrep` format.

This script is looking to mine data for _a single timestep_ of the dyncore. It will
jump over the first timestep and record the second one.

The database is mined for:
* CUDA kernel cumulative time, median time, calls
* Halo exchange exact time (if the `nsys` runs was done using the `external_profiler --nvtx` option)
The script will print results and/or write CSV & plots.

Usage: `nsys_mine.py <--csv> <--plots> <NSYS_GENERATED.sqlite>`
* `<--csv>` will write down in .csv the results
* `<--plots>` will plot the results (cumulative time, counts, median time, cumulative/call)

## TODO list:

* json configuration
* record all versions of a stencil then `exit()` from the program
