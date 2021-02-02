#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} ${THRESH_ARGS}"

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    make tests_venv_mpi
else
    make tests_mpi
fi
