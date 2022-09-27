#!/bin/bash
set -e -x
BACKEND=$1
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make physics_savepoint_tests_mpi
else
    DEV=n make physics_savepoint_tests_mpi
fi
