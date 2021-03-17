# How to get performance numbers

## Daint

### Arguments

- timesteps: Number of timesteps to execute (this includes the first one as a warm up step)
- ranks: Number of ranks to run with
- backend: choice of gt4py backend
- data_path: the test data

### Constraints

The data directory is expected to be serialized data (serialized by serialbox). The archive `dat_files.tar.gz` gets unpacked. The serialized data is also expected to have both the `input.nml` as well as the `*.yml` namelists present.

### Output

A `timing.json` file containing statistics over the ranks for execution time. The first timestep is counted towards `init`, the rest of the timesteps are in `main loop`. Total is inclusive of the other categories

### Example

`examples/standalone/benchmarks/run_on_daint.sh 60 6 gtx86`
