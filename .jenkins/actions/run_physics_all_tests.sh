#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../"

if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${JENKINS_DIR}/${XML_REPORT}"
    CONTAINER_CMD="srun" make physics_savepoint_tests && CONTAINER_CMD="" make physics_savepoint_tests_mpi make physics_savepoint_tests_mpi
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/${XML_REPORT}"
    make physics_savepoint_tests && make physics_savepoint_tests_mpi
fi
