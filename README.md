
# fv3gfs-integration

FV3GFS-integration is the top level directory that includes the FV3 dynamical core, physics, and util.

**WARNING** This repo is under active development and relies on code and data that is not publicly available at this point.

## Getting started

If you want to build the main fv3gfs-integration docker image, run

 ```shell
$ make build
```

If you want to download test data, run

```shell
$ make get_test_data
```

To enter the docker image, run

```shell
$ make dev
```

Currently, we only support tests in the dynamical core and util. More details can be found in the respective folder. 