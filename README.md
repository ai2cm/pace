FV3ser
======
FV3ser ('ef-vee-threezer') is a Python (using Gt4py with Dawn) version of the FV3 dynamical core (fv3gfs-fortran repo), with regression test data of computation units coming from serialized output from the fortran model generated with the Gridtools/serialbox repo.

------------

Relevant repositories:
https://github.com/VulcanClimateModeling/fv3gfs-fortran
https://github.com/VulcanClimateModeling/serialbox2 fork of https://github.com/GridTools/serialbox.git
https://github.com/GridTools/gt4py
https://github.com/MeteoSwiss-APN/dawn

------------
Checking out
------------

This package uses submodules. After you check out the repository, you must run
`git submodule init` followed by `git submodule update` in the root directory of this package.

------------
Installation
------------



-- build a container with required dependencies for running the python code
make build

-- unit testing -- will grab regression data if it doesn't exist locally
make run_unit_tests

-- unit testing with options
make tests TEST_ARGS="add pytest command line args (see below)"

Test options:
   --which_modules <modules to run tests for> : comma separated list of which modules to test (default 'all')
   
   --print_failures : if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data

   --failure_stride: whhen printing failures, print avery n failures only
   
   --data_path : path to where you have the Generator*.dat and *.json serialization regression data. Defaults to current directory.
   
   --data_backend : which backend to use for data storage, defulat: numpy, other options: gtmc, gtx86, gtcuda, debug
   
   --exec_backend: which backend to use for stencil computation, default numpy, other options: gtmc, gtx86, gtcuda, debug, and dawn:gtmc

-- to just get the regression data

If testing/developing new test code:
docker run -v <Local fv3gfs checkout>/fv3_python_port/:/port_dev -v /Volumes/Dev/devcode/runfv3scripts/input_data/:/test_data   --name port -it  us.gcr.io/vcm-ml/fv3gfs-gt4p
Then in the container :
pytest -v -s --data_path="/test_data/" /fv3_python_port/fv3/test
Change test code, try again, etc.

------------
Generating test data
------------
pip install -r external/fv3gfs-fortran/requirements.txt