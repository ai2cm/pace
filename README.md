# Pace

Pace is an implementation of the FV3GFS / SHiELD atmospheric model developed by NOAA/GFDL using the GT4Py domain-specific language in Python. The model can be run on a laptop using Python-based backend or on thousands of heterogeneous compute nodes of a large supercomputer.

The top level directory contains the FV3 dynamical core (fv3core), the GFS physics package (fv3gfs-physics), and infrastructure utilities (pace-util).

**WARNING** This repo is under active development and supported features and procedures can change rapidly and without notice.

This git repository is laid out as a mono-repo, containing multiple independent projects. Because of this, it is important not to introduce unintended dependencies between projects. The graph below indicates a project depends on another by an arrow pointing from the parent project to its dependency. For example, the tests for fv3core should be able to run with only files contained under the fv3core and util projects, and should not access any files in the driver or physics packages. Only the top-level tests in Pace are allowed to read all files.

![Graph of interdependencies of Pace modules, generated from dependences.dot](./dependencies.svg)

## Testing all components

Before the top-level build, make sure you have configured the authentication with user credientials and configured Docker with the following commands:
```shell
gcloud auth login
gcloud auth configure-docker
```

You will also need to update the git submodules are cloned and at the correct version:
```shell
git submodule update --init --recursive
```

Then build `pace` docker image at the top level.

```shell
make build
```

## Dynamical core tests

To run dynamical core tests, first get the test data from inside the `fv3core` directory.

```shell
cd fv3core
make get_test_data
cd ../
```

For sequential tests (these take a bit of time), there are two options:

1. To enter the container and run the dynamical core sequential tests (main and savepoint tests):

```shell
make dev
cd /pace
pytest -v -s --data_path=/pace/fv3core/test_data/8.1.1/c12_6ranks_standard/dycore/ ./fv3core/tests
```

2. To run the tests without opening the docker container (just savepoint tests):

```shell
DEV=y make savepoint_tests
```

For parallel tests:

1. Enter the container and run the dynamical core parallel tests

```shell
make dev
cd /pace
mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/pace/fv3core/test_data/c12_6ranks_standard/ ./fv3core/tests
```

2. Run the tests without opening the docker container

```shell
DEV=y make savepoint_tests_mpi
```

Additional test options are described under `fv3core` documentation.

## Physics tests

To run physics tests, first get the test data from inside the `fv3core` directory. Currently only the microphysics is supported.

```shell
cd fv3gfs-physics 
make get_test_data
cd ..
```

In the container, the sequential physics tests can be run by:

```shell
DEV=y make dev
cd /pace
pytest -v -s --data_path=/pace/fv3gfs-physics/test_data/8.1.1/c12_6ranks_baroclinic_dycore_microphysics/physics/ ./fv3gfs-physics/tests --threshold_overrides_file=/pace/fv3gfs-physics/tests/savepoint/translate/overrides/baroclinic.yaml
```
In this case, DEV=y mounts the local directory, so any changes in it will take effect without needing to rebuild the container.

or use the second method (as in dynamical core testing) outside of the docker container:

```shell
DEV=y make physics_savepoint_tests
```

For running the parallel tests use:

```shell
make dev
cd /pace
mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/pace/fv3gfs-physics/test_data/8.1.1/c12_6ranks_baroclinic_dycore_microphysics/physics/ ./fv3gfs-physics/tests --threshold_overrides_file=/pace/fv3gfs-physics/tests/savepoint/translate/overrides/baroclinic.yaml
```

or

```shell
DEV=y make physics_savepoint_tests_mpi
```

## Infrastructure utilities tests

Inside the container, the infrastructure utilities tests can be run as follows:

```shell
make dev
cd /pace/pace-util
make test
```
