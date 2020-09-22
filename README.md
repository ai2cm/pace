# FV3core

FV3core is a Python version, using GT4Py with CPU and GPU backend options, of the FV3 dynamical core (fv3gfs-fortran repo).
The code here includes regression test data of computation units coming from serialized output from the Fortran model generated using the `GridTools/serialbox` framework.


## Getting started

Use the `tests` target of the Makefile to run the unit tests.

```shell
$ make tests
```
This will pull the test data from the Google storage bucket (using the `make get_test_data`) if it does not exist locally yet, build the fv3core docker image, and run all of the sequential tests using that image.

See the [Unit Testing]() section below for options.

If you'd like to run MPI parallel tests (which are needed for parts of the code with halo updates), run

```shell
$ make tests_mpi
```

To rebuild the backend Dawn/GT4Py environment before running tests, either run

```shell
$ PULL=False make tests
```

```shell
$ make rebuild_environment
$ make tests
```

### Test data options

If you want to run different test data, discover the possible options with
```shell
$ make list_test_data_options

This will list the storage buckets in the cloud. Then to run one of them, set EXPERIMENT to the folder name of the data you'd like to use:
e.g.
```shell
$EXPERIMENT=c48_6ranks_standard make tests
```
If you choose an experiment with a different number of ranks than 6, also set `NUM_RANKS=<num ranks>`

## Running tests  inside a container

If you to prefer to work interactively inside the fv3core container, get the test data and build the docker image:
```shell
$ make get_test_data
```

```shell
$ make build
```
Testing can be run with this data from `/port_dev` inside the container:

```shell
$ make dev
```

Then in the container:

```shell
$ pytest -v -s --data_path=/test_data/ /port_dev/tests --which_modules=<stencil name>
```
The 'stencil name' can be determined from the associated Translate class. e.g. TranslateXPPM is a test class that translate data serialized from a run of the fortran model, and 'XPPM' is the name you can use with --which_modules.


## Testing interactively outside the container

After `make tests` has been run at least once (or you have data in test_data and the docker image fv3core exists because `make build` has been run), you can iterate on code changes using

```shell
$ make dev_tests
```
or for the parallel tests:

```shell
$ make dev_tests_mpi
```
These will mount your current code into the fv3core container and run it rather than the code that was built when `make build` ran.


### Test options

All of the make endpoints involved running tests can be prefixed with the `TEST_ARGS` environment variable to set test options or pytest CLI args (see below) when running inside the container.

* `--which_modules <modules to run tests for>` - comma separated list of which modules to test (default 'all').

* `--print_failures` - if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data.

* `--failure_stride` - whhen printing failures, print avery n failures only.

* `--data_path` - path to where you have the `Generator*.dat` and `*.json` serialization regression data. Defaults to current directory.

* `--backend` - which backend to use for the computation. Defaults to numpy. Other options: gtmc, gtcuda, dawn:gtmc.

Pytest provides a lot of options, which you can see by `pytest --help`. Here are some
common options for our tests, which you can add to `TEST_ARGS`:

* `-r` - is used to report test types other than failure. It can be provided `s` for skipped (e.g. tests which were not run because earlier tests of the same stencil failed), `x` for xfail or "expected to fail" tests (like tests with no translate class), or `p` for pass. For example, to report skipped and xfail tests you would use `-rsx`.

* `--disable-warnings` - will stop all warnings from being printed at the end of the tests, for example warnings that translate classes are not yet implemented.

* `-v` - will increase test verbosity, while `-q` will decrease it.

* `-s` - will let stdout print directly to console instead of capturing the output and printing it when a test fails only. Note that logger lines will always be printed both during (by setting log_cli in our pytest.ini file) and after tests.

* `-m` - will let you run only certain groups of tests. For example, `-m=parallel` will run only parallel stencils, while `-m=sequential` will run only stencils that operate on one rank at a time.


## Porting a new stencil

1. Find the location in the fv3gfs-fortran repo code where the save-point is to be added, e.g. using

```shell
$ git grep <stencil_name> <checkout of fv3gfs-fortran>
```

2. Create a `translate` class from the serialized save-point data to a call to the stencil or function that calls the relevant stencil(s).

These are usually named `tests/translate/translate_<lowercase name>`

Import this class in the `tests/translate/__init__.py` file

3. Write a Python function wrapper that the translate function (created above) calls.

By convention, we name these `fv3core/stencils/<lower case stencil name>.py`

4. Run the test, either with one name or a comma-separated list

```shell
$ make dev_tests TEST_ARGS="-–which_modules=<stencil name(s)>"
```


## Installation

To build the `us.gcr.io/vcm-ml/fv3core` image with required dependencies for running the Python code, run

```shell
$ make build
```

Add `PULL=False` to build from scratch without running `docker pull`:

```shell
PULL=False make build
```

## Relevant repositories

- https://github.com/VulcanClimateModeling/serialbox2 -
  Serialbox generates serialized data when the Fortran model runs and has bindings to manage data from Python

- https://github.com/VulcanClimateModeling/fv3gfs-fortran -
  This is the existing Fortran model decorated with serialization statements from which the test data is generated


- https://github.com/GridTools/gt4py -
  Python package for the DSL language

- https://github.com/MeteoSwiss-APN/dawn -
  DSL language compiler using the GridTools parallel execution model

Some of these are submodules.
While tests can work without these, it may be necessary for development to have these as well.
To add these to the local repository, run

```shell
$ git submodule update --init --recursive
```

The submodule include:

- `external/fv3util` - git@github.com:VulcanClimateModeling/fv3util.git





## Dockerfiles

There are three main driver files:

1. `docker/Dockerfile.build_environment` - builds off of the serialbox environment from fv3gfs-fortran, installs Serialbox, Dawn and GT4Py

2. `docker/Dockerfile` - uses the build environment and copies in the fv3 folder only. This is to make development easier so that when you change a file in fv3, 'make build' does not accidentally or otherwise trigger a 20 minute rebuild of all of those installations, but just updates the code in the fv3core image.



## Linting

Before committing your code, you can automatically fix many common style linting errors by running

```shell
$ make reformat
```

To list linting issues

```shell
$ make lint
```

## License

FV3Core is provided under the terms of the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
