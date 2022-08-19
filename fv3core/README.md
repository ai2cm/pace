> DISCLAIMER: Work in progress

# FV3core

FV3core is a Python version, using GridTools GT4Py with CPU and GPU backend options, of the FV3 dynamical core (fv3gfs-fortran repo).
The code here includes regression test data of computation units coming from serialized output from the Fortran model generated using the `GridTools/serialbox` framework.

As of January 10, 2021 this documentation is outdated in that it was written when we had fv3core as its own single repository. Some functionality, such as linting, has been moved to the top level but may still be described in this document as occuring inside the fv3core folder.

**WARNING** This repo is under active development and relies on code and data that is not publicly available at this point.

## QuickStart

1. Ensure you have docker installed and available for building and running and has access to the VCM cloud

Be sure to complete any required post-installation instructions (e.g. [for linux](https://docs.docker.com/engine/install/linux-postinstall/)). Also [authorize Docker to pull from gcr](https://cloud.google.com/container-registry/docs/advanced-authentication). Your user will need to have read access to the `us.gcr.io/vcm-ml` repository.

2.  You can build the image, download the data, and run the tests using:

```shell
$ make tests savepoint_tests savepoint_tests_mpi
```

If you want to develop code, you should also install the linting requirements and git hooks locally

```shell
$ pip install -c constraints.txt -r requirements/requirements_lint.txt
$ pre-commit install

## Getting started, in more detail
If you want to build the main fv3core docker image, run

```shell
$ make build
```

If you want to download test data run

```shell
$ make get_test_data
```

And the c12_6ranks_standard data will download into the `test_data` directory.

If you do not have a GCP account, there is an option to download basic test data from a public FTP server and you can skip the GCP authentication step above. To download test data from the FTP server, use `make USE_FTP=yes get_test_data` instead and this will avoid fetching from a GCP storage bucket. You will need a valid in stallation of the `lftp` command.

MPI parallel tests (that run that way to exercise halo updates in the model) can also be run with:

```shell
$ make savepoint_tests_mpi
```

The environment image that the fv3core container uses is prebuilt and lives in the GCR. The above commands will by default pull this image before building the fv3core image and running the tests.
To build the environment from scratch (including GT4py) before running tests, either run

```
make build_environment
```

or

```shell
$ PULL=False make savepoint_tests
```

which will execute the target `build_environment` for you before running the tests.

There are `push_environment` and `rebuild_environment` targets, but these should normally not be done manually. Updating the install image should only be done by Jenkins after the tests pass using a new environment.

### Test data options

If you want to run different test data, discover the possible options with
```shell
$ make list_test_data_options
```
This will list the storage buckets in the cloud. Then to run one of them, set EXPERIMENT to the folder name of the data you'd like to use:

e.g.
```shell
$EXPERIMENT=c48_6ranks_standard make tests
```

If you choose an experiment with a different number of ranks than 6, also set `NUM_RANKS=<num ranks>`

## Testing interactively outside the container

After `make savepoint_tests` has been run at least once (or you have data in test_data and the docker image fv3core exists because `make build` has been run), you can iterate on code changes using

```shell
$ DEV=y make savepoint_tests
```
or for the parallel or non-savepoint tests:

```shell
$ DEV=y make tests savepoint_tests_mpi
```
These will mount your current code into the fv3core container and run it rather than the code that was built when `make build` ran.

## Running tests inside a container

If you to prefer to work interactively inside the fv3core container, get the test data and build the docker image (see above if you do not have a GCP account and want to get test data):
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

* `--backend` - which backend to use for the computation. Options: `[numpy, gt:cpu_ifirst, gt:cpu_first, gt:gpu, cuda]`. Defaults to `numpy`.
* `--python_regression` - Run the tests that have Python based regression data. Only applies to running parallel tests (savepoint_tests_mpi)
Pytest provides a lot of options, which you can see by `pytest --help`. Here are some
common options for our tests, which you can add to `TEST_ARGS`:

* `-r` - is used to report test types other than failure. It can be provided `s` for skipped (e.g. tests which were not run because earlier tests of the same stencil failed), `x` for xfail or "expected to fail" tests (like tests with no translate class), or `p` for pass. For example, to report skipped and xfail tests you would use `-rsx`.

* `--disable-warnings` - will stop all warnings from being printed at the end of the tests, for example warnings that translate classes are not yet implemented.

* `-v` - will increase test verbosity, while `-q` will decrease it.

* `-s` - will let stdout print directly to console instead of capturing the output and printing it when a test fails only. Note that logger lines will always be printed both during (by setting log_cli in our pytest.ini file) and after tests.

* `-m` - will let you run only certain groups of tests. For example, `-m=parallel` will run only parallel stencils, while `-m=sequential` will run only stencils that operate on one rank at a time.

* `--threshold_overrides_file` - will read a yaml file with error thresholds specified for specific backend and platform (docker or metal) configurations, overriding the max_error thresholds defined in the Translate classes. Format of the yaml file is described [here](tests/savepoint/translate/overrides/README.md).

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

These are usually named `tests/savepoint/translate/translate_<lowercase name>`

Import this class in the `tests/savepoint/translate/__init__.py` file

3. Write a Python function wrapper that the translate function (created above) calls.

By convention, we name these `fv3core/stencils/<lower case stencil name>.py`

4. Run the test, either with one name or a comma-separated list

```shell
$ make dev_tests TEST_ARGS="-–which_modules=<stencil name(s)>"
```

**Please also review the [Porting conventions](#porting-conventions) section for additional explanation**
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

- https://github.com/VulcanClimateModeling/pace-util
  Python specific model functionality, such as halo updates.

- https://github.com/VulcanClimateModeling/fv3gfs-wrapper
  A Python based wrapper for running the Fortran version of the FV3GFS model.

Some of these are submodules.
While tests can work without these, it may be necessary for development to have these as well.
To add these to the local repository, run

```shell
$ git submodule update --init
```

The submodules include:

- `external/pace-util` - git@github.com:VulcanClimateModeling/pace-util.git
- `external/daint_venv` -  git@github.com:VulcanClimateModeling/daint_venv.git

## Dockerfiles and building

There are two main docker files:

1. `docker/dependencies.Dockerfile` - defines dependency images such as for mpi, serialbox, and GT4py

2. `docker/Dockerfile` - uses the dependencies to define the final fv3core images.

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

## Pinned dependencies

Dependencies are pinned using `constraints.txt`. This is auto-generated by pip-compile from the `pip-tools` package, which reads `requirements.txt` and `requirements/requirements_lint.txt`, determines the latest versions of all dependencies (including recursive dependencies) compatible those files, and writes pinned versions for all dependencies. This can be updated using:

```shell
$ make constraints.txt
```

This file is committed to the repository, and gives more reproducible tests if an old commit of the repository is checked out in the future. The constraints are followed when creating the `fv3core` docker images. To ensure consistency this should ideally be run from inside a docker development environment, but you can also run it on your local system with an appropriate Python 3 environment.

## Development

To develop fv3core, you need to install the linting requirements in `requirements/requirements_lint.txt`. To install the pinned versions, use:

```shell
$ pip install -c constraints.txt -r requirements/requirements_lint.txt
```

This adds `pre-commit`, which we use to lint and enforce style on the code. The first time you install `pre-commit`, install its git hooks using:

```shell
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

As a convenience, the `lint` target of the top-level makefile executes `pre-commit run --all-files`.
Linting, which formats files and checks for some style conventions, is required, as the same checks are the first step in the continuous integration testing that happens when creating a pull request.
Linting locally saves time and literal energy, since CI tests do not have to be launched so many times!

 Please see the 'Development Guidelines' below for more information on the structure of the code to align your new code with the current conventions, as well as the CONTRIBUTING.md document for style guidelines.

## GT4Py version

FV3Core does not actually use the [GridTools/gt4py](https://github.com/gridtools/gt4py) main, it instead uses a Vulcan Climate Modeling development branch.
This is publically available version at [VCM/gt4py](https://github.com/vulcanclimatemodeling/gt4py).

Situation: There is a new stable feature in a gt4py PR, but it is not yet merged into the GridTools/gt4py main branch.
[branches.cfg](https://github.com/VulcanClimateModeling/gt4py/blob/develop/branches.cfg) lists these features.
Steps:

1. Add any new branches to `branches.cfg`
2. Rebuild the develop branch, either:
  a. `make_develop gt4py-dev path/to/branches.cfg` (you may have to resolve conflicts...)
  b. Adding new commits on top of the existing develop branch (e.g. merge or cherry-pick)
3. Force push to the develop branch: `git push -f upstream develop`

The last step will launch Jenkins tests. If these pass:

1. Create a git tag: `git tag v-$(git rev-parse --short HEAD)`
2. Push the tag: `git push upstream --tags`
3. Make a PR to [VCM/gt4py](https://github.com/vulcanclimatemodeling/fv3core) that updates the version in `docker/Makefile` to the new tag.

## License
FV3Core is provided under the terms of the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.

# Development guidelines

## File structure / conventions
The main functionality of the FV3 dynamical core, which has been ported from the Fortran version in the fv3gfs-fortran repo, is defined using GT4py stencils and python 'compute' functions in fv3core/stencils. The core is comprised of units of calculations defined for regression testing. These were initially generally separated into distinct files in fv3core/stencils with corresponding files in tests/savepoint/translate/translate_<unit>.py defining the translation of variables from Fortran to Python. Exceptions exist in cases where topical and logical grouping allowed for code reuse. As refactors optimize the model, these units may be merged to occupy the same files and even methods/stencils, but the units should still be tested separately, unless determined to be redundant.

The core has most of its calculations happening in GT4py stencils, but there are still several instances of operations happening in Python directly, which will need to be replaced with GT4py code for optimal performance.

The namelist and grid are global variables defined in fv3core/_config.py The namelist is 'flattened' so that the grouping name of the option is not required to access the data (we may want to change this).

The grid variables are mostly 2d variables and are 'global' to the model thread per mpi rank. The grid object also contains domain and layout information relevant to the current rank being operated on.

Utility functions in `fv3core/utils/` include:
  - `gt4py_utils.py`:
    - default gt4py and model settings
    - methods for generating gt4py storages
    - methods for using numpy and cupy arrays in python functions that have not been put into GT4py
    - methods for handling complex patterns that did not immediately map to gt4py, and will mostly be removed with future refactors (e.g. k_split_run)
    - some general model math computations (e.g. great_circle_dist), that will eventually be put into gt4py with a future refactor
  - `grid.py`:
    - A Grid class definition that provides information about the grid layout, current tile informationm access to grid variables used globally, and convenience methods related to tile indexing, origins and domains commonly used
    - A grid is defined for each MPI rank (minimum 6 ranks, 1 for each tile face of the cubed sphere grid represnting the whole Earth)
    - Also provides functionality for generating a Quantity object used for halo updates and other utilities
  - `corners`: port of corner calculations, initially direct Python calculations, being replaced with GT4py gtscript functions as the GT4py regions feature is implemented
  - `mpi.py`: a wrapper for importing mpi4py when available
  - `global_constants.py`: constants for use throughout the model
  - `typing.py`: Clean names for common types we use in the model. This is new and
    hasn't been adopted throughout the model yet, but will eventually be our
    standard. A shorthand 'sd' has been used in the intial version.

The `tests/` directory currently includes a framework for translating fields serialized (using
Serialbox from GridTools) from a Fortran run into gt4py storages that can be inputs to
fv3core unit computations, and compares the results of the ported code to serialized
data following a unit computation.

The `docker/` directory provides Dockerfiles for building a repeatable environment in which
to run the core

The `external/` directory is for submoduled repos that provide essential functionality

The build system uses Makefiles following the convention of other repos within VulcanClimateModeling.

## Model Interface

The top level functions fv_dynamics and fv_sugridz can currenty only be run in parallel using mpi with a minimum of 6 ranks (there are a few other units that also require this, e.g. whenever there is a halo update involved in a unit). These are the interface to the rest of the model and currently have different conventions than the rest of the model.
 - A 'state' object (currently a SimpleNamespace) stores pointers to the allocated data fields
 - Most functions within dyn_core can be run sequentially per rank
 - Currently a list of ArgSpecs must decorate an interface function, where each ArgSpec provides useful information about the argument, e.g.: `@state_inputs( ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout")`
   - The format is (fortran_name, long_name, units, intent)
   - We currently provide a duplicate of most of the metadata in the specification of the unit test, but that may be removed eventually.
 - Then the function itself, e.g. fv_dynamics, has arguments of 'state', 'comm' (the communicator) and all of the scalar parameters being provided.

### Porting conventions

Generation of regression data occurs in the fv3gfs-fortran repo (https://github.com/VulcanClimateModeling/fv3gfs-fortran) with serialization statements and a build procedure defined in `tests/serialized_test_data_generation`. The version of data this repo currently tests against is defined in `FORTRAN_SERIALIZED_DATA_VERSION` in this repo's `docker/Makefile.image_names`. Fields serialized are defined in Fortran code with serialization comment statements such as:

```
    !$ser savepoint C_SW-In
    !$ser data delpcd=delpc delpd=delp ptcd=ptc
```

where the name being assigned is the name the fv3core uses to identify the variable in the test code. When this name is not equal to the name of the variable, this was usually done to avoid conflicts with other parts of the code where the same name is used to reference a differently sized field.

The majority of the logic for translating from data serialized from Fortran to something that can be used by Python, and the comparison of the results, is encompassed by the main Translate class in the tests/savepoint/translate/translate.py file. Any units not involving a halo update can be run using this framework, while those that need to be run in parallel can look to the ParallelTranslate class as the parent class in tests/savepoint/translate/parallel_translate.py. These parent classes provide generally useful operations for translating serialized data between Fortran and Python specifications, and for applying regression tests.

A new unit test can be defined as a new child class of one of these, with a naming convention of `Translate<Savepoint Name>` where `Savepoint Name` is the name used in the serialization statements in the Fortran code, without the `-In` and `-Out` part of the name. A translate class can usually be minimally specify the input and output fields. Then, in cases where the parent compute function is insuffient to handle the complexity of either the data translation or the compute function, the appropriate methods can be overridden.

For Translate objects
  - The init function establishes the assumed translation setup for the class, which can be dynamically overridden as needed.
  - the parent compute function does:
    - Makes gt4py storages of the max shape (grid.npx+1, grid.npy+1, grid.npz+1) aligning the data based on the start indices specified. (gt4py requires data fields have the same shape, so in this model we have buffer points so all calculations can be done easily without worrying about shape matching).
    - runs the compute function (defined in self.compute_func) on the input data storages
    - slices the computed Python fields to be compared to fortran regression data
  - The unit test then uses a modified relative error metric to determine whether the unit passes
  - The init method for a Translate class:
    - The input (self.in_vars["data_vars"]) and output(self.out_vars) variables are specified in dictionaries, where the keys are the name of the variable used in the model and the values are dictionaries specifying metadata for translation of serialized data to gt4py storages. The metadata that can be specied to override defaults are:
    - Indices to line up data arrays into gt4py storages (which all get created as the max possible size needed by all operations, for simplicity): "istart", "iend", "jstart", "jend", "kstart", "kend". These should be set using the 'grid' object available to the Translate object, using equivalent index names as in the declaration of variables in the Fortran code, e.g. real:: cx(bd%is:bd%ie+1,bd%jsd:bd%jed ) means we should assign. Example:

```python
      self.in_vars["data_vars"]["cx"] = {"istart": self.is\_, "iend": self.ie + 1,
                                         "jstart": self.jsd, "jend": self.jed,}
```
  - There is only a limited set of Fortran shapes declared, so abstractions defined in the grid can also be used,
    e.g.: `self.out_vars["cx"] = self.grid.x3d_compute_domain_y_dict()`. Note that the variables, e.g. `grid.is\_` and `grid.ie` specify the 'compute' domain in the x direction of the current tile, equivalent to `bd%is` and `bd%ie` in the Fortran model EXCEPT that the Python variables are local to the current MPI rank (a subset of the tile face), while the Fortran values are global to the tile face. This is because these indices are used to slice into fields, which in Python is 0-based, and in Fortran is based on however the variables are declared. But, for the purposes of aligning data for computations and comparisons, we can match them in this framework. Shapes need to be defined in a dictionary per variable including `"istart"`, `"iend"`, `"jstart"`, `"jend"`, `"kstart"`, `"kend"` that represent the shape of that variable as defined in the Fortran code. The default shape assumed if a variable is specified with an empty dictionary is `isd:ied, jsd:jed, 0:npz - 1` inclusive, and variables that aren't that shape in the Fortran code need to have the 'start' indices specified for the in_vars dictionary , and 'start' and 'end' for the out_vars.
    - `"serialname"` can be used to specify a name used in the Fortran code declaration if we'd like the model to use a different name
    - `"kaxis"`: which dimension is the vertical direction. For most variables this is '2' and does not need to be specified. For Fortran variables that assign the vertical dimension to a different axis, this can be set to ensure we end up with 3d storages that have the vertical dimension where it is expected by GT4py.
    - `"dummy_axes"`: If set this will set of the storage to have singleton dimensions in the axes defined. This is to enable testing stencils where the full 3d data has not been collected and we want to run stencil tests on the data for a particular slice.
    - `"names_4d"`: If a 4d variable is being serialized, this can be set to specify the names of each 3d field. By default this is the list of tracers.
    - input variables that are scalars should be added to `self.in_vars["parameters"]`
    - `self.compute_func` is the name of the model function that should be run by the compute method in the translate class
    - `self.max_error` overrides the parent classes relative error threshold. This should only be changed when the reasons for non-bit reproducibility are understood.
    - `self.max_shape` sets the size of the gt4py storage created for testing
    - `self.ignore_near_zero_errors[<varname>] = True`: This is an option to let some fields pass with higher relative error if the absolute error is very small

For `ParallelTranslate` objects:
  - Inputs and outputs are defined at the class level, and these include metadata such as the "name" (e.g. understandable name for the symbol), dimensions, units and n_halo(numb er of halo lines)
  - Both `compute_sequential` and `compute_parallel` methods may be defined, where a mock communicator is used in the `compute_sequential` case
  - The parent assumes a state object for tracking fields and methods exist for translating from inputs to a state object and extracting the output variables from the state. It is assumed that Quantity objects are needed in the model method in order to do halo updates.
  - `ParallelTranslate2Py` is a slight variation of this used for many of the parallel units that do not yet utilize a state object and relies on the specification of the same index metadata of the Translate classes
  - `ParallelTranslateBaseSlicing` makes use of the state but relies on the Translate object of self._base, a Translate class object, to align the data before making quantities, computing and comparing.

### Debugging Tests

Pytest can be configured to give you a pdb session when a test fails. To route this properly through docker, you can run:

```bash
TEST_ARGS="-v -s --pdb" RUN_FLAGS="--rm -it" make tests
```

This can be done with any pytest target, such as `make savepoint_tests` and `make savepoint_tests_mpi`.
