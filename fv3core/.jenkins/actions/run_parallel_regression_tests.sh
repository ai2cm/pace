#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="${EXTRA_TEST_ARGS} -v -s -rsx --backend=${BACKEND} "

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" make savepoint_tests_mpi
else
    make savepoint_tests_mpi
fi
export TEST_ARGS="${TEST_ARGS} --compute_grid"
if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" make savepoint_tests_mpi
else
    if [[ ${FV3_DACEMODE} == "True" ]]; then
        RUN_FLAGS="--rm -e FV3_DACEMODE=True" make savepoint_tests_mpi
    else
        make savepoint_tests_mpi
    fi
fi
