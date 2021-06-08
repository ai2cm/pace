#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" make savepoint_tests_mpi
else
    make savepoint_tests_mpi
fi
