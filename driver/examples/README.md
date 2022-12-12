# Driver Examples

Here we have example scripts and configuration for running the Pace driver.
Currently this contains two examples which run the model on a baroclinic test case using the numpy backend.
You will find this runs fairly slowly, since the "compiled" code is still Python.
In the future, we will add examples for the compiled backends we support.

The "docker" workflow here is written for the basic baroclinic test example.
`write_then_read.sh` is the second example, which shows how to configure an MPI run to write communication data to disk and then use it to repeat a single rank of that run with the same configuration.
This second example assumes you are already in an appropriate environment to run the driver, for example as documented in the "Host Machine" section below.

Note that on the baroclinic test case example you will see significant grid imprinting in the first hour time evolution.
Rest assured that this duplicates the behavior of the original Fortran code.

We have also included a utility to convert the zarr output of the run to netcdf, for convenience. To convert `output.zarr` to `output.nc`, you would run:

```bash
$ python3 zarr_to_nc.py output.zarr output.nc
```

Another example is `baroclinic_init.py`, which initializes a barcolinic wave and writes out the grid and the initial state. To run this script with the c12 6ranks example:

```bash
$ mpirun -n 6 python3 baroclinic_init.py ./configs/baroclinic_c12.yaml
```
## Docker

To run a baroclinic c12 case with Docker in a single command, run `run_docker.sh`.
This example will start from the Python 3.8 docker image, install extra dependencies and Python packages, and execute the example, leaving the output in this directory.

To visualize the output, two example scripts are provided:
1. `plot_output.py`: To use it, you must install matplotlib (e.g. with `pip install matplotlib`).
2. `plot_cube.py`: this uses plotting tools in [fv3viz](https://github.com/ai2cm/fv3net/tree/master/external/fv3viz). Note the requirements aren't part of pace by default and need to be installed accordingly. It is recommended to use the post processing docker provided at the top level `docker/postprocessing.Dockerfile`.

## Host Machine

To run examples on your host machine, you will need to have an MPI library on your machine suitable for installing mpi4py.
For example, on Ubuntu 20.04 this could be the libopenmpi3 and libopenmpi-dev libraries.

With these requirements installed, set up the virtual environment with

```bash
$ create_venv.sh
$ . venv/bin/activate
```

With the environment activated, the model itself can be run with `python3 -m pace.driver.run <config yaml>`.
Currently this must be done using some kind of mpirun, though there are plans to enable single-tile runs without MPI.
The exact command will vary based on your MPI implementation, but you can try running

```bash
$ mpirun -n 6 python3 -m pace.driver.run ./configs/baroclinic_c12.yaml
```

To run the example at C48 resolution instead of C12, you can update the value of `nx_tile` in the configuration file from 12 to 48.
Here you can also change the timestep in seconds `dt_atmos`, as well as the total run duration with `minutes`, or by adding values for `hours` or `days`.
