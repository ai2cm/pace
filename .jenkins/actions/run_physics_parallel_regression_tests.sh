#!/bin/bash
set -e -x
BACKEND=$1
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

if [ ${python_env} == "virtualenv" ]; then
    CONTAINER_CMD="" make physics_savepoint_tests_mpi
else
    make physics_savepoint_tests_mpi
fi
