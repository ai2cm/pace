To generate the unit testing docker image

-- build a container with serialbox2 installed
make build_ serialize

-- Generate serialization data
make run_serialize

-- build the container with the unit testing code
 make build_unit_testing

-- Once you have serialization data locally
make run_unit_tests

If you prefer to run manually
docker run  -v "/<local_path_to_serialized_data>":/test_data --rm us.gcr.io/vcm-ml/fv3gfs-gt4py pytest -v -s --data_path="/test_data/" /fv3_python_port/fv3/test

Test options:
   --which_modules <modules to run tests for> : comma separated list of which modules to test (default 'all')
   
   --print_failures : if your test fails, it will only report the first datapoint. If you want all the nonmatching regression data to print out (so you can see if there are patterns, e.g. just incorrect for the first 'i' or whatever'), this will print out for every failing test all the non-matching data
   
   --data_path : path to where you have the Generator*.dat and *.json serialization regression data. Defaults to current directory.


If testing/developing new test code:
docker run -v <Local fv3gfs checkout>/fv3_python_port/:/port_dev -v /Volumes/Dev/devcode/runfv3scripts/input_data/:/test_data   --name port -it  us.gcr.io/vcm-ml/fv3gfs-gt4p
Then in the container :
pytest -v -s --data_path="/test_data/" /fv3_python_port/fv3/test
Change test code, try again, etc. 