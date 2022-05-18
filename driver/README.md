# pace-driver

This package provides command-line routines to run the Pace model, and utilities to write model driver scripts.

We suggest reading the code in the examples directory, or taking a look at `pace/driver/run.py` to see how the main entrypoint for this package works.

# Usage

Usage examples exist in the examples directory.
The command-line interface may be run in certain debugging modes in serial, but usually you will want to run it using an mpi executor such as mpirun.

```bash
$ python3 -m pace.driver.run --help
Usage: python -m pace.driver.run [OPTIONS] CONFIG_PATH

  Run the driver.

  CONFIG_PATH is the path to a DriverConfig yaml file.

Options:
  --log-rank INTEGER  rank to log from, or all ranks by default, ignored if
                      running without MPI
  --log-level TEXT    one of 'debug', 'info', 'warning', 'error', 'critical'
  --help              Show this message and exit.
```

A DriverConfig yaml file is the yaml representation of a DriverConfig object, which can be found in the code of this module.
