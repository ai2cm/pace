# How to get performance numbers

## Daint

### Arguments

-   Timesteps: Number of timesteps to execute (this includes the first one as a warm up step)
-   Ranks: Number of ranks to run with
-   backend: choice of gt4py backend
-   (target directory): the output timing.json file goes here
-   (data directory): the test data

### Constraints

The data directory is expected to be serialized data (serialized by serialbox). The archive `dat_files.tar.gz` gets unpacked. The serialized data is also expected to have both the `input.nml` as well as the `*.yml` namelists present.

### Output

a `timing.json` file containing statistics over the ranks for execution time. The first timestep is counted towards `init`, the rest of the timesteps are in `main loop`. Total is inclusive of the other categories

### Example

`examples/standalone/benchmarks/run_on_daint.sh 60 6 gtx86`

## Local Performance

### Arguments

-   data dir: the test data
-   timestep: number of timesteps
-   backend: chose the backend

### Constraints

The data directory is expected to be serialized data (serialized by serialbox). The serialized data is expected to be already unpacked with two files `input.yml` as well as `input.nml` as the namelist in there

### Output

a `timing.json` file containing statistics over the ranks for execution time. The first timestep is counted towards `init`, the rest of the timesteps are in `main loop`. Total is inclusive of the other categories

### Example

`examples/standalone/runfile/dynamics.py test_data/ 60 gtx86`
