#!/bin/bash
set -e -x
BACKEND=$1

export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

if [ ${python_env} == "virtualenv" ]; then
    TARGET=driver CONTAINER_CMD="" make driver_savepoint_tests_mpi
else
    TARGET=driver make driver_savepoint_tests_mpi
fi
