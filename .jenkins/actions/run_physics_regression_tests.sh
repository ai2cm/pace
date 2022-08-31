#!/bin/bash
set -e -x
BACKEND=$1
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../"
if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${JENKINS_DIR}/${XML_REPORT}"
    CONTAINER_CMD="srun" DEV=n make physics_savepoint_tests
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/${XML_REPORT}"
    DEV=n make physics_savepoint_tests
fi
