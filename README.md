# Pace

Pace is the top level directory that includes the FV3 dynamical core, physics, and util.

If you are visiting for AMS 2022, we recommend you go to `driver/README.md`.

**WARNING** This repo is under active development and relies on code and data that is not publicly available at this point.


Currently, we support tests in the dynamical core, physics, and util.

This git repository is laid out as a mono-repo, containing multiple independent projects. Because of this, it is important not to introduce unintended dependencies between projects. The graph below indicates a project depends on another by an arrow pointing from the parent project to its dependency. For example, the tests for fv3core should be able to run with only files contained under the fv3core and util projects, and should not access any files in the driver or physics packages. Only the top-level tests in Pace are allowed to read all files.

![Graph of interdependencies of Pace modules, generated from dependences.dot](./dependencies.svg)

## Dynamical core tests

Before the top-level build, make sure you have configured the authentication with user credientials and configured Docker with the following commands:
```shell
$ gcloud auth login
$ gcloud auth configure-docker
```

You will also need to update the git submodules are cloned and at the correct version:
```shell
$ git submodule update --init --recursive
```

To run dynamical core tests, first get the test data from inside `fv3core` or `fv3gfs-physics` folder, then build `fv3gfs-integration` docker image at the top level.

```shell
$ cd fv3core
$ make get_test_data
$ cd ../
$ make build
```

For serial tests (these take a bit of time), there are two options:

(1) To enter the container and run the dynamical core serial tests (main and savepoint tests):

```shell
$ make dev
$ pytest -v -s --data_path=/fv3core/test_data/8.1.0/c12_6ranks_standard/dycore/ /fv3core/tests
```

(2) To run the tests without opening the docker container (just savepoint tests):

```shell
$ DEV=y make savepoint_tests
```

For parallel tests:

```shell
$ mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/fv3core/test_data/c12_6ranks_standard/ /fv3core/tests
```

or

```shell
$ DEV=y make savepoint_tests_mpi
```

Additional test options are described under `fv3core` documentation.

## Physics tests

Currently, the supported test case is dynamical core + microphysics: e.g., `c12_6ranks_baroclinic_dycore_microphysics` (gs://vcm-fv3gfs-serialized-regression-data/integration-7.2.5/c12_6ranks_baroclinic_dycore_microphysics).

To download the data and open the Docker container:

```shell
$ cd fv3gfs-physics
$ make get_test_data
$ cd ..
```

In the container, physics tests can be run by:

```shell
$ DEV=y make dev
$ pytest -v -s --data_path=/test_data/8.1.0/c12_6ranks_baroclinic_dycore_microphysics/physics/ /fv3gfs-physics/tests --threshold_overrides_file=/fv3gfs-physics/tests/savepoint/translate/overrides/baroclinic.yaml
```
In this case, DEV=y mounts the local directory, so any changes in it will take effect without needing to rebuild the container.

or use the second method (as in dynamical core testing) outside of the docker container:

```shell
$ DEV=y make physics_savepoint_tests
```
----------------------

For parallel tests use:

```shell
$ mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/test_data/8.1.0/c12_6ranks_baroclinic_dycore_microphysics/physics/ /fv3gfs-physics/tests --threshold_overrides_file=/fv3gfs-physics/tests/savepoint/translate/overrides/baroclinic.yaml
```

or

```shell
$ DEV=y make physics_savepoint_tests_mpi
```

--------

## Util tests

Inside the container, util tests can be run from `/pace-util`:

```shell
$ cd /pace-util
$ make test
```
