#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="${EXTRA_TEST_ARGS} -v -s -rsx --backend=${BACKEND} "

# sync the test data
make get_test_data

export CPPFLAGS="${CPPFLAGS} -Wno-unused-but-set-variable"

if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make savepoint_tests_mpi
    TARGET=init CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make savepoint_tests_mpi
else
    DEV=n make savepoint_tests_mpi
    TARGET=init DEV=n make savepoint_tests_mpi
fi
export TEST_ARGS="${TEST_ARGS} --compute_grid"
if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make savepoint_tests_mpi
    TARGET=init CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make savepoint_tests_mpi
else
    DEV=n make savepoint_tests_mpi
    TARGET=init DEV=n make savepoint_tests_mpi
fi
