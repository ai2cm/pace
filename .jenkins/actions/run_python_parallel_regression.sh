#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} ${THRESH_ARGS}"
# sync the test data
make get_test_data
export TEST_ARGS="${ARGS} --which_modules=FVSubgridZ"
if [ ${python_env} == "virtualenv" ]; then
  make tests_venv_mpi
else
  make tests_mpi
fi

set +e
export TEST_ARGS="${ARGS} --python_regression"
if [ ${python_env} == "virtualenv" ]; then
    make tests_venv_mpi
else
    make tests_mpi
fi
if [ $? -ne 0 ] ; then
    echo "PYTHON REGRESSIONS failed, looking for errors in the substeps:"
    set -e
    export TEST_ARGS="${ARGS}"
    if [ ${python_env} == "virtualenv" ]; then
	make tests_venv
	make tests_venv_mpi
    else
	make tests
	make tests_mpi
    fi
    exit 1
fi
set -e
exit 0
