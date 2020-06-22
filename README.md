# FV3ser

FV3ser (pronounced *ef-vee-threezer*) is a Python version, using GT4Py with Dawn, of the FV3 dynamical core (fv3gfs-fortran repo).
The code here includes regression test data of computation units coming from serialized output from the Fortran model generated using the `GridTools/serialbox` framework.


## Getting started

Use the `tests` target of the Makefile to run the unit tests

```shell
$ make tests
```

See the [Unit Testing]() section below for options.

This will create a running Docker container `TestDataContainer-<version>`, from which you can create a volume for the test data.

```shell
$ make cleanup_container
```
will stop and remove the unit testing container.

To rebuild the backend Dawn/GT4Py environment before running tests, either run

```shell
$ PULL=False make tests
```

```shell
$ make rebuild_environment
$ make tests
```

If you'd like to run MPI parallel tests, as opposed to parallel tests run sequentially, run

```shell
$ make tests_mpi
```


## Porting a new stencil

1. Find the location in code where the save-point is to be added, e.g. using

```shell
$ git grep <stencil_name>
```

2. Create a `translate` class from the serialized save-point data to a call to the stencil or function that calls the relevant stencil(s).

These are usually named `fv3/translate/translate_<lowercase name>`

3. Write a Python function wrapper that the translate function (created above) calls.

By convention, we name these `fv3/stencils/<lower case stencil name>.py`

4. Run the test, either with one name or a comma-separated list

```shell
$ pytest -–which_modules=<stencil name(s)>
```


## Developing stencils

Changes to code in the `fv3/` directory can be validated by running all tests using `make test`.

Alternatively, you can develop stencils using data and code as volumes into the container.
Two approaches:

### Using test data in a container

```shell
$ make pull_test_data
$ make devc
```

### Extracting test data

If you want to work with test data outside the container, set the env variable `TEST_DATA_HOST` (by default `./test-data`) to a location, then run

```shell
$ make pull_test_data
$ make extract_test_data
```

After extracting the test data, testing can be run with this data from `/port_dev` inside the container:

```shell
$ (cd /port_dev && make dev)
```

This process can be done without the Makefile target using

```shell
$ docker run -v <path/to/fv3gfs>:/port_dev -v <test data path>:/test_data --name <your favorite name> -it us.gcr.io/vcm-ml/fv3ser
```

Then in the container:

```shell
$ pytest -v -s --data_path=/test_data/ /port_dev/fv3/test --which_modules=<stencil name>
```


## Installation


To build the `us.gcr.io/vcm-ml/fv3ser` image with required dependencies for running the Python code, run

```shell
$ make build
```

Add `PULL=False` to build from scratch without running `docker pull`:

```shell
PULL=False make build
```


## Relevant repositories

- https://github.com/VulcanClimateModeling/fv3gfs-fortran -
  This is the existing Fortran model decorated with serialization statements

- https://github.com/VulcanClimateModeling/serialbox2 -
  Serialbox generates serialized data when the Fortran model runs and has bindings to manage data from Python

- https://github.com/VulcanClimateModeling/fv3config -
  FV3Config is used to configure and manipulate run directories for FV3GFS

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

- `external/fv3gfs-fortran` - git@github.com:VulcanClimateModeling/fv3gfs-fortran
- `external/fv3config` - https://github.com/VulcanClimateModeling/fv3config
- external/fv3gfs-python` - git@github.com:VulcanClimateModeling/fv3gfs-python.git


## Unit testing stencils

To run regression tests of the Python FV3 functions and compare to serialized data from the Fortran model run any of:

```shell
$ make tests # sets up test data before running
$ make run_tests_container # simply runs tests
$ make run_tests_host_data # uses data at TEST_DATA_HOST
```

All of these can be prefixed with the `TEST_ARGS` environment variable to set test options or pytest CLI args (see below).

### Test options

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


## Generating test data

This should be done hopefully infrequently in conjunction with a change to the serialization statements in `fv3gfs-fortran`

1. Make changes to fv3gfs-fortran, PR and merge changes to master, or if just testing an idea, `git submodule update --init --recursive` to work on the fortran code locally.

2. Increment the `FORTRAN_VERSION` constant in the Makefile

3. `make generate_test_data` -- this will
    1. Run `git submodule update` if you have not already
	  2. Compile the fortran model with serialization on
	  2. Generate a run directory using fv3config in a container (also submoduled) and the 'fv3config.yml' configuration specified in fv3/test
	  4. Run the model on this configuration
	  5. Copy the data to a new image and delete the rundirectory image (it is large and usually don't need it anymore)

4. If you want to make this change permanent, then open a PR, merge, and run `post_test_data`


## Dockerfiles

There are three main driver files:

1. `docker/Dockerfile.build_environment` - builds off of the serialbox environment from fv3gfs-fortran, installs Dawn and GT4Py

2. `docker/Dockerfile` - uses the build environment and copies in the fv3 folder only. This is to make development easier so that when you change a file in fv3, 'make build' does not accidentally or otherwise trigger a 20 minute rebuild of all of those installations, but just updates the code in the fv3ser image.

3. `docker/Dockerfile.fortran_model_data` - builds the fv3gfs-fortran model with serialization on, sets up a run directory and generates test data. This is to be done infrequently and is orthogonal to the content in the other dockerfiles.


## Linting

Before committing your code, you can automatically fix many common style linting errors by running

```shell
$ make reformat
```

To list linting issues

```shell
$ make lint
```
