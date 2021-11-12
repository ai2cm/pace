#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} "
# sync the test data
make get_test_data
export TEST_ARGS="${ARGS} --which_modules=FVSubgridZ"
if [ ${python_env} == "virtualenv" ]; then
    export CONTAINER_CMD=""
fi

set +e
export TEST_ARGS="${ARGS} --python_regression"
make savepoint_tests_mpi

if [ $? -ne 0 ] ; then
    echo "PYTHON REGRESSIONS failed, looking for errors in the substeps:"
    set -e
    export TEST_ARGS="${ARGS}"
    make savepoint_tests
    make savepoint_tests_mpi
    exit 1
fi
set -e
exit 0
