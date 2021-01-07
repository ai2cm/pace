#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}
# Set the host data location
export TEST_DATA_HOST="${TEST_DATA_DIR}/${EXPNAME}/"
# sync the test data
make get_test_data
make run_tests_parallel TEST_ARGS="${ARGS} --which_modules=FVSubgridZ"
set +e
make run_tests_parallel TEST_ARGS="${ARGS} --python_regression"
if [ $? -ne 0 ] ; then
    echo "PYTHON REGRESSIONS failed, looking for errors in the substeps:"
    set -e
    make run_tests_sequential TEST_ARGS="${ARGS}"
    make run_tests_parallel TEST_ARGS="${ARGS}"
    exit 1
fi
set -e
exit 0
