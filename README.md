FV3ser
======
FV3ser ('ef-vee-threezer') is a Python (using Gt4py with Dawn) version of the FV3 dynamical core (fv3gfs-fortran repo), with regression test data of computation units coming from serialized output from the fortran model generated with the Gridtools/serialbox repo.


Getting started
---------------
* To run the existing unit tests use
`make tests`.
See 'Unit Testing' section below for unit test options.
This will create a running container TestDataContainer-<version> which can be used to volume the test data from.
`make cleanup_container` will stop and remove this container.

* To rebuild the backend Dawn/Gt4py environment before running tests run
`PULL=False make tests`
OR
`make rebuild_environment` followed by
`make tests`

If you'd like to run mpi parallel tests (as opposed to parallel tests run sequentially),
`make tests_mpi`

Porting a new stencil
---------------------

1. git grep the name of the stencil in ``fv3gfs-fortran`` to find the place in code
   where the save-point is added
2. create a class that translates from the serialized save-point data to a call
   to the stencil (or function that calls the relevant stencil(s) -- fv3/translate/translate_<lowercase name> 
3. write a python function the translate function calls that does the calculation of interest,
   in fv3/stencils/<lower case stencil name>.py 
4. test using ``pytest -–which_modules=<stencil name>``, either with one name or a comma-separated list


Developing stencils
-------------------
Make a code change in the 'fv3' directory, then run 'make tests'
OR
Option: develop stencils using data and code as volumes into a container, there are a couple of possibilities:
1. Using test data in a container:
make pull_test_data
'make devc'
2. Extracting test data:


* To extract test data from the container if you want to work with it outside the container
make pull_test_data
make extract_test_data
After you have extracted the test data:
'make dev'
Then run the tests from /port_dev instead of /fv3

If you prefer using docker run directly:
docker run -v <Local fv3gfs checkout>:/port_dev -v <TEST DATA PATH>:/test_data   --name <your favorite name> -it us.gcr.io/vcm-ml/fv3ser

Then in the container:

pytest -v -s --data_path=/test_data/ /port_dev/fv3/test --which_modules=<Your stencil>


Installation
------------

-- build the us.gcr.io/vcm-ml/fv3ser container with required dependencies for running the python code

make build

To build from scratch (without docker pulling)
PULL=False make build


Relevant repositories
---------------------
This package uses submodules that can be used to generate serialization data. If you would like to work with them (e.g. to make new serialization data etc.), after you check out the repository, you can run
`git submodule update --init --recursive`

https://github.com/VulcanClimateModeling/fv3gfs-fortran
https://github.com/VulcanClimateModeling/fv3config
https://github.com/VulcanClimateModeling/serialbox2 fork of https://github.com/GridTools/serialbox.git
https://github.com/GridTools/gt4py
https://github.com/MeteoSwiss-APN/dawn



'Unit' testing stencils
-----------------------
How to run regression tests of python FV3 functions and comparing to data serialized from the fortran model
TEST_ARGS="add pytest command line args (see below)" make tests (or run_tests_container or run_tests_host_data)

Test options:
   --which_modules <modules to run tests for> : comma separated list of which modules to test (default 'all')

   --print_failures : if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data

   --failure_stride: whhen printing failures, print avery n failures only

   --data_path : path to where you have the `Generator*.dat` and `*.json` serialization regression data. Defaults to current directory.

   --backend : which backend to use for the computation. Defaults to numpy. Other options: gtmc, gtcuda, dawn:gtmc

Pytest provides a lot of options, which you can see with `pytest --help`. Here are some
common options for our tests, which you can add to `TEST_ARGS`:

- `-r` is used to report test types other than failure. It can be provided `s` for
  skipped (e.g. tests which were not run because earlier tests of the same stencil
  failed), `x` for xfail or "expected to fail" tests (like tests with no translate
  class), or `p` for pass. For example, to report skipped and xfail tests you would
  use `-rsx`.
- `--disable-warnings` will stop all warnings from being printed at the end of the tests,
  for example warnings that translate classes are not yet implemented
- `-v` will increase test verbosity, while `-q` will decrease it
- `-s` will let stdout print directly to console instead of capturing the output and
  printing it when a test fails only. Note that logger lines will always be printed
  both during (by setting log_cli in our pytest.ini file) and after tests.
- `-m` will let you run only certain groups of tests. For example, `-m=parallel` will
  run only parallel stencils, while `-m=sequential` will run only stencils that operate
  on one rank at a time


Generating test data
--------------------
* This should be done hopefully infrequently in conjunction with a change to the serialization statements in fv3gfs-fortran
1. make changes to fv3gfs-fortran, PR and merge changes to master (or if just testing an idea, 'git submodule update --init --recursive' to work on the fortran code locally)
2. increment the FORTRAN_VERSION in the Makefile
3. 'make generate_test_data' -- this will
        * git submodule update if you have not already
	* compile the fortran model with serialization on
	* generate a run directory using fv3config in a container (also submoduled) and the 'fv3config.yml' configuration specified in fv3/test
	* run the model on this configuration
	* copy the data to a new image and delete the rundirectory image (it is large and usually don't need it anymore)
4. if you want to commit this, open a PR, merge and run 'post_test_data'


Dockerfiles
-----------

There are 3 main Dockerfiles in the 'docker' folder
 1) Dockerfile.build_environment -- builds off of the serialbox environment from fv3gfs-fortran, installs Dawn and Gt4py
 2) Dockerfile -- uses the build environment and copies in the fv3 folder only. This is to make development easier so that when you change a file in fv3, 'make build' does not accidentally or otherwise trigger a 20 minute rebuild of all of those installations, but just updates the code in the fv3ser image.
 3) Dockerfile.fortran_model_data -- builds the fv3gfs-fortran model with serialization on, sets up a run directory and generates test data. This is to be done infrequently and is orthogonal to the content in the other dockerfiles.


Linting
-------

Before committing your code, you can automatically fix many common style linting errors
with `make reformat`. `make lint` will tell you about any remaining issues which need
to be fixed manually.
