# Translate Test Guide

First, make sure you have followed the instruction in the top level [README](../../../../README.md) to install the Python requirements, GT4Py, and Pace.

## Downloading test data

The unit and regression tests of pace require data generated from the Fortran reference implementation which has to be downloaded from a Google Cloud Platform storage bucket. Since the bucket is setup as "requester pays", you need a valid GCP account to download the test data.

First, make sure you have configured the authentication with user credientials and configured Docker with the following commands:
```shell
gcloud auth login
gcloud auth configure-docker
```

Next, you can download the test data for the dynamical core and the physics tests.

```shell
cd $(git rev-parse --show-toplevel)/fv3core
make get_test_data
cd $(git rev-parse --show-toplevel)/physics
make get_test_data
```

If you do not have a GCP account, there is an option to download basic test data from a public FTP server and you can skip the GCP authentication step above. To download test data from the FTP server, use `make USE_FTP=yes get_test_data` instead and this will avoid fetching from a GCP storage bucket. You will need a valid in stallation of the `lftp` command.

## Running the tests (manually)

There are two ways to run the tests, manually by explicitly invoking `pytest` or autmatically using make targets. The former can be used both inside the Docker container as well as for a bare-metal installation and will be described here.

First enter the container and navigate to the pace directory:

```shell
cd $(git rev-parse --show-toplevel)
make dev
cd /pace
```

Note that by entering the container with the `make dev` command, volumes for code and test data will be mounted into the container and modifications inside the container will be retained.

There are two sets of tests. The "sequential tests" test components which do not require MPI-parallelism. The "parallel tests" can only within an MPI environment.

To run the sequential and parallel tests for the dynmical core (fv3core), you can execute the following commands (these take a bit of time):

```shell
pytest -v -s --data_path=/pace/fv3core/test_data/8.1.1/c12_6ranks_standard/dycore/ ./fv3core/tests
mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/pace/fv3core/test_data/8.1.1/c12_6ranks_standard/dycore ./fv3core/tests
```

Note that you must have already downloaded the test data according to the instructions above. The precise path needed for `--data_path` may be different, particularly the version directory.

Similarly, you can run the sequential and parallel tests for the physical parameterizations (physics). Currently, only the microphysics is integrated into pace and will be tested.

```shell
pytest -v -s --data_path=/pace/test_data/8.1.1/c12_6ranks_baroclinic_dycore_microphysics/physics/ ./physics/tests --threshold_overrides_file=/pace/physics/tests/savepoint/translate/overrides/baroclinic.yaml
mpirun -np 6 python -m mpi4py -m pytest -v -s -m parallel --data_path=/pace/test_data/8.1.1/c12_6ranks_baroclinic_dycore_microphysics/physics/ ./physics/tests --threshold_overrides_file=/pace/physics/tests/savepoint/translate/overrides/baroclinic.yaml
```

Finally, to test the pace infrastructure utilities (util), you can run the following commands:

```shell
cd $(git rev-parse --show-toplevel)/util
make test
make test_mpi
```

## Running the tests automatically using Docker

To automatize testing, a set of convenience commands is available that build the Docker image, run the container and execute the tests (dynamical core and physics only). This is mainly useful for CI/CD workflows.

```shell
cd $(git rev-parse --show-toplevel)
DEV=y make savepoint_tests
DEV=y make savepoint_tests_mpi
DEV=y make physics_savepoint_tests
DEV=y make physics_savepoint_tests_mpi
```
