
# fv3gfs-integration

FV3GFS-integration is the top level directory that includes the FV3 dynamical core, physics, and util.

**WARNING** This repo is under active development and relies on code and data that is not publicly available at this point.

## Getting started

Currently, we only support tests in the dynamical core and util. 

### Dynamical core tests

To run dynamical core tests, first get the test data from inside `fv3core` or `fv3gfs-physics` folder, then build `fv3gfs-integration` docker image at the top level.

```shell
$ cd fv3core
$ make get_test_data
$ cd ../
$ make build
```

To enter the container:
```shell
$ make dev
```

Then in the container, dynamical core serial tests can be run:

```shell
$ pytest -v -s --data_path=/port_dev/fv3core/test_data/c12_6ranks_standard/ /port_dev/fv3core/tests
```

For parallel tests:

```shell
$ mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/port_dev/fv3core/test_data/c12_6ranks_standard/ /port_dev/fv3core/tests
```

Additional test options are described under `fv3core` documentation.

### Util tests

Inside the container, util tests can be run from `/port_dev`:
```shell
$ cd /port_dev
$ make test_util 
```