FV3ser
======
FV3ser ('ef-vee-threezer') is a Python (using Gt4py with Dawn) version of the FV3 dynamical core (fv3gfs-fortran repo), with regression test data of computation units coming from serialized output from the fortran model generated with the Gridtools/serialbox repo.

------------
Getting started
------------
* To run the existing unit tests without rebuilding docker containers
'PULL=True make tests'  -- this will pull the existing docker image for the repo and run the tests
See 'Unit Testing' section below for unit test options

------------
Extract test data from the container if you want to work with it outside the container
------------
make pull_test_data
make_extract_test_data

------------
Developing stencils
------------
It may be easier to volume in the current directory into a container and run tests there (though you can also make build, make tests if you prefer)
After you have extracted the test data:
'make dev'
The run the tests from /port_dev instead of /fv3

If you prefer using docker run directly:
docker run -v <Local fv3gfs checkout>:/port_dev -v <TEST DATA PATH>:/test_data   --name <your favorite name> -it us.gcr.io/vcm-ml/fv3py
Then in the container :
pytest -v -s --data_path=/test_data/ /port_dev/fv3/test --which_modules=<Your stencil>

------------
Installation
------------

-- build the us.gcr.io/vcm-ml/fv3py container with required dependencies for running the python code (rather than docker pulling it)
make build

------------
Relevant repositories
------------
This package uses submodules that can be used to generate serialization data. If you would like to work with them (e.g. to make new serialization data etc.), after you check out the repository, you can run
`git submodule update --init --recursive`

https://github.com/VulcanClimateModeling/fv3gfs-fortran
https://github.com/VulcanClimateModeling/fv3config
https://github.com/VulcanClimateModeling/serialbox2 fork of https://github.com/GridTools/serialbox.git
https://github.com/GridTools/gt4py
https://github.com/MeteoSwiss-APN/dawn


------------
'Unit' testing stencils -- running regression tests of python FV3 functions and comparing to data serialized from the fortran model
------------
TEST_ARGS="add pytest command line args (see below)" make tests (or run_tests_container or run_tests_host_data)

Test options:
   --which_modules <modules to run tests for> : comma separated list of which modules to test (default 'all')
   
   --print_failures : if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data

   --failure_stride: whhen printing failures, print avery n failures only
   
   --data_path : path to where you have the Generator*.dat and *.json serialization regression data. Defaults to current directory.
   
   --data_backend : which backend to use for data storage, defulat: numpy, other options: gtmc, gtx86, gtcuda, debug
   
   --exec_backend: which backend to use for stencil computation, default numpy, other options: gtmc, gtx86, gtcuda, debug, and dawn:gtmc
