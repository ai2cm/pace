> DISCLAIMER: Work in progress

# FV3core

FV3core is a Python version, using GridTools GT4Py with CPU and GPU backend options, of the FV3 dynamical core (fv3gfs-fortran repo).
The code here includes regression test data of computation units coming from serialized output from the Fortran model generated using the `GridTools/serialbox` framework.

**WARNING** This repo is under active development and relies on code and data that is not publicly available at this point.

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

The environment image that the fv3core container uses is prebuilt and lives in the GCR. The above commands will by default pull this image before building the fv3core image and running the tests.
To build the environment from scratch (including GT4py) before running tests, either run

```
make build_environment
```

or


```shell
$ PULL=False make tests
```

which will execute the target `build_environment` for you before running the tests.

There are `push_environment` and `rebuild_environment` targets, but these should normally not be done manually. Updating the install image should only be done by Jenkins after the tests pass using a new environment.

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

## Running tests inside a container

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




### Test options

All of the make endpoints involved running tests can be prefixed with the `TEST_ARGS` environment variable to set test options or pytest CLI args (see below) when running inside the container.

* `--which_modules <modules to run tests for>` - comma separated list of which modules to test (defaults to running all of them).

* `--print_failures` - if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data.

* `--failure_stride` - when printing failures, print every n failures only.

* `--data_path` - path to where you have the `Generator*.dat` and `*.json` serialization regression data. Defaults to current directory.

* `--backend` - which backend to use for the computation. Defaults to numpy. Other options: gtmc, gtcuda, gtx86.

Pytest provides a lot of options, which you can see by `pytest --help`. Here are some
common options for our tests, which you can add to `TEST_ARGS`:

* `-r` - is used to report test types other than failure. It can be provided `s` for skipped (e.g. tests which were not run because earlier tests of the same stencil failed), `x` for xfail or "expected to fail" tests (like tests with no translate class), or `p` for pass. For example, to report skipped and xfail tests you would use `-rsx`.

* `--disable-warnings` - will stop all warnings from being printed at the end of the tests, for example warnings that translate classes are not yet implemented.

* `-v` - will increase test verbosity, while `-q` will decrease it.

* `-s` - will let stdout print directly to console instead of capturing the output and printing it when a test fails only. Note that logger lines will always be printed both during (by setting log_cli in our pytest.ini file) and after tests.

* `-m` - will let you run only certain groups of tests. For example, `-m=parallel` will run only parallel stencils, while `-m=sequential` will run only stencils that operate on one rank at a time.

**NOTE:** FV3 is current assumed to be by default in a "development mode", where stencils are checked each time they execute for code changes (which can trigger regeneration). This process is somewhat expensive, so there is an option to put FV3 in a performance mode by telling it that stencils should not automatically be rebuilt:

```shell
$ export FV3_STENCIL_REBUILD_FLAG=False
```

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

- https://github.com/GridTools/serialbox -
  Serialbox generates serialized data when the Fortran model runs and has bindings to manage data from Python

- https://github.com/VulcanClimateModeling/fv3gfs-fortran -
  This is the existing Fortran model decorated with serialization statements from which the test data is generated

- https://github.com/GridTools/gt4py -
  Python package for the DSL language

- https://github.com/VulcanClimateModeling/fv3gfs-util
  Python specific model functionality, such as halo updates.

- https://github.com/VulcanClimateModeling/fv3gfs-wrapper
  A Python based wrapper for running the Fortran version of the FV3GFS model.

Some of these are submodules.
While tests can work without these, it may be necessary for development to have these as well.
To add these to the local repository, run

```shell
$ git submodule update --init --recursive
```

The submodule include:

- `external/fv3gfs-util` - git@github.com:VulcanClimateModeling/fv3gfs-util.git


## Dockerfiles and building

There are two main docker files:

1. `docker/dependencies.Dockerfile` - defines dependency images such as for fv3gfs-fortran, serialbox, and GT4py

2. `docker/Dockerfile` - uses the dependencies to define the final fv3core and fv3core-wrapper images.

The dependencies are separated out into their own images to expedite rebuilding the docker image without having to rebuild dependencies, especially on CI.

For the commands below using `make -C docker`, you can alternatively run `make` from within the `docker` directory.

These dependencies can be updated, pushed, and pulled with `make -C docker build_deps`, `make -C docker push_deps`, and `make -C docker pull_deps`. The tag of the dependencies is based on the tag of the current build in the Makefile, which we will expand on below.

Building from scratch requires both a deps and build command, such as `make -C docker pull_deps fv3core_image`.

If any example fails for "pulled dependencies", it means the dependencies have never been built. You can
build them and push them to GCR with:

```shell
$ make -C docker build_deps push_deps
```

### Building examples

fv3core image with pulled dependencies:

```shell
$ make -C docker pull_deps fv3core_image
```

CUDA-enabled fv3core image with pulled dependencies:
```
$ CUDA=y make -C docker pull_deps fv3core_image
```

fv3core image with locally-built dependencies:
```shell
$ make -C docker build_deps fv3core_image
```

fv3core-wrapper image with pulled dependencies:

```shell
$ make -C docker pull_deps fv3core_wrapper_image
```

CUDA-enabled fv3core-wrapper image with pulled dependencies:
```
$ CUDA=y make -C docker pull_deps fv3core_wrapper_image
```

## Running with fv3gfs-wrapper

To use the python dynamical core for model runs, use the fv3core-wrapper image.

A development environment for fv3gfs-wrapper is set up under the make target:

```shell
$ make dev_wrapper
```

This will bind-mount in fv3core and the submodules in `external` such as `external/fv3gfs-fortran`,
`external/fv3gfs-wrapper`, and `external/fv3gfs-util` and compile your bind-mounted sources upon
entering. If you change the fortran code while in the development environment, you need to re-compile
the fortran code and then re-build the wrapper for your changes to be reflected. You can do this
from the root of the image using:

```shell
$ make -C external/fortran install && make -C external/fv3gfs-wrapper build
```

to install fv3core as an importable module. Alternatively, you can specify `develop` instead of `install` if you want to edit the fv3core code.

### Updating Serialbox

If you need to install an updated version of Serialbox, you must first install cmake into the development environment. To install an updated version of Serialbox from within the container run

```shell
$ wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz && \
  tar xzf cmake-3.17.3.tar.gz && \
  cd cmake-3.17.3 && \
  ./bootstrap && make -j4 && make install
$ git clone -b v2.6.1 --depth 1 https://github.com/GridTools/serialbox.git /tmp/serialbox
$ cd /tmp/serialbox
$ cmake -B build -S /tmp/serialbox -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/serialbox
$ cmake --build build/ -j $(nproc) --target install
$ cd -
$ rm -rf build /tmp/serialbox
```

### Running a wrapper runfile

To set up a model run, the `write_run_directory` command will create a rundir containing the needed inputs and structure for the model run based on a configuration yaml file:

```shell
$ write_run_directory path/to/configuration/yaml path/to/rundir
```

A few example config files are provided in the `fv3config` repository and `fv3core/examples/wrapped/config`. After the rundir has been created you can link or copy a runfile such as `fv3core/examples/wrapped/runfiles/fv3core_test.py` to your rundir and run it with mpirun:

```shell
$ mpirun -np X python fv3core_test.py
```

## Pinned dependencies

Dependencies are pinned using `constraints.txt`. This is auto-generated by pip-compile from the `pip-tools` package, which reads `requirements.txt` and `requirements_lint.txt`, determines the latest versions of all dependencies (including recursive dependencies) compatible those files, and writes pinned versions for all dependencies. This can be updated using:

```shell
$ make constraints.txt
```

This file is committed to the repository, and gives more reproducible tests if an old commit of the repository is checked out in the future. The constraints are followed when creating the `fv3core` and `fv3core_wrapper` docker images. To ensure consistency this should ideally be run from inside a docker development environment, but you can also run it on your local system with an appropriate Python 3 environment.

## Development

To develop fv3core, you need to install the linting requirements in `requirements_lint.txt`. To install the pinned versions, use:

```shell
$ pip install -r requirements_lint.txt -c constraints.txt
```

This adds `pre-commit`, which we use to lint and enforce style on the code. The first time you install `pre-commit`, install its git hooks using:

```shell
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

As a convenience, the `lint` target of the top-level makefile executes `pre-commit run --all-files`.

## License

FV3Core is provided under the terms of the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
